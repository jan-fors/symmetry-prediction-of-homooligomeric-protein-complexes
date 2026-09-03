import os
from urllib import request, error
from pathlib import Path
import argparse
import requests
import re
import json

# HTTP connections
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "PDB-Symmetry-Search/1.0"
})

def get_json(url: str) -> dict:
    """
    GET JSON from RCSB API.

    Args:
        url (str): URL to fetch

    Returns:
        dict: parsed JSON response
    """
    response = SESSION.get(url, timeout=60)

    if response.status_code == 204:
        return {}

    response.raise_for_status()
    return response.json()

def normalize_stoichiometry(stoichiometry) -> str:
    """
    Convert RCSB stoichiometry field to a clean string.

    Args:
        stoichiometry (str or list): RCSB stoichiometry field

    Returns:
        str: normalized stoichiometry string
    """

    if stoichiometry is None:
        return ""

    if isinstance(stoichiometry, list):
        if len(stoichiometry) == 0:
            return ""

        # Common case: ["A2"]
        if len(stoichiometry) == 1:
            return str(stoichiometry[0]).strip()

        # Fallback: join multiple entries
        return "".join(str(x).strip() for x in stoichiometry)

    return str(stoichiometry).strip()

def is_homomeric_stoichiometry(stoichiometry) -> bool:
    """
    Check whether stoichiometry looks homomeric.
    Kept:
        A, A2, A3, A12, A24, A60
    Excluded:
        AB, A2B2, ABC, A3B3

    Args:
        stoichiometry (str): RCSB stoichiometry field

    Returns:
        bool: True if homomeric, False otherwise
    """

    stoichiometry = normalize_stoichiometry(stoichiometry)

    if not stoichiometry or stoichiometry in [".", "?"]:
        return False

    return re.fullmatch(r"A\d*", stoichiometry) is not None


def fetch_bioassemblies(rcsb_id: str, outdir: Path):
    """
    Script from Laura P.

    Download all bioassembly files for given PDB entry and saves the paths as attribute

    Parameters:
        outdir (str): Directory where the files are stored

    Returns:
        list: List of file paths
    """
    assembly_id = 1
    id_specific_output = outdir / Path(rcsb_id)
    os.makedirs(id_specific_output, exist_ok=True)

    while True:
        url = f"https://files.rcsb.org/download/{rcsb_id}-assembly{assembly_id}.cif.gz"
        file_path = f"{id_specific_output}/{rcsb_id}-assembly{assembly_id}.cif.gz"

        

        try:
            # Download the file
            request.urlretrieve(url, file_path)
           
            # symmetry
            url = "https://data.rcsb.org/rest/v1/core/assembly/{pdb_id}/{assembly_id}".format(
                        pdb_id=rcsb_id.lower(),
                        assembly_id=assembly_id,
                    )
            
            assembly_json = get_json(url)

            # Extract asymmetric unit IDs for the assembly
            assembly_infos = assembly_json.get("pdbx_struct_assembly_gen", [])
            asym_ids_assembly = []
            for assembly_info in assembly_infos:
                current_assembly_id = assembly_info.get("assembly_id")
                if not current_assembly_id == assembly_id:
                    continue
                asym_ids_assembly.extend(assembly_info.get("asym_id_list", []))
                
            # Extract symmetry information from the assembly JSON
            symmetry_rows = assembly_json.get("rcsb_struct_symmetry", [])
    
            if isinstance(symmetry_rows, dict):
                symmetry_rows = [symmetry_rows]
    
            for sym in symmetry_rows:
                kind = sym.get("kind", "")
                symbol = sym.get("symbol", "")
    
                # Keep global symmetry annotation
                if kind and kind != "Global Symmetry":
                    continue
    
                stoichiometry = normalize_stoichiometry(
                    sym.get("stoichiometry")
                    or sym.get("global_stoichiometry")
                    or assembly_json.get("rcsb_assembly_info", {}).get("stoichiometry")
                    or ""
                )
                global_symmetry = (
                    symbol
                    or sym.get("global_symmetry")
                    or ""
                )
    
                # Check if the assembly is homomeric based on stoichiometry
                is_homomeric = is_homomeric_stoichiometry(stoichiometry)

                data = {
                    "rcsb_id": rcsb_id,
                    "assembly": assembly_id,
                    "stoichiometry": stoichiometry,
                    "symmetry": global_symmetry,
                    "kind": kind,
                    "homomeric": is_homomeric
                }
                print(data)
                json_path = f"{id_specific_output}/{rcsb_id}-assembly{assembly_id}.json"
                json_str = json.dumps(data, indent=4)
                with open(json_path, "w") as f:
                    f.write(json_str)

            assembly_id += 1

        except error.HTTPError as e:
            if e.code == 404:
                break  # Stop if 404 (Not Found) error occurs
            else:
                print(f"HTTP error: {e}")
                break  # Stop on other HTTP errors

        except error.URLError as e:
            print(f"URL error: {e}")
            break  # Stop on connection failure
        
        except Exception as e:
            print(e)
            break





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download all assemblies connected to one rcsb id."
    )
    parser.add_argument("rcsb_id", type=str)
    parser.add_argument("-o", "--output_dir", type=Path, default=".")
    args = parser.parse_args()

    fetch_bioassemblies(args.rcsb_id, args.output_dir)
