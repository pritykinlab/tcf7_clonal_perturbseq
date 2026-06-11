import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

class GOParser:
    """
    Parses a JSON file containing GO term information.
    """
    def __init__(self, json_file=REPO_ROOT / 'data/misc/msigdb_GO_pathways.json'):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def find_genes_by_GO_number(self, go_number):
        for key, value in self.data.items():
            if value['exactSource'] == go_number:
                return value['geneSymbols']
        return None

    def find_collection_by_GO_number(self, go_number):
        for key, value in self.data.items():
            if value['exactSource'] == go_number:
                return value['collection']
        return None

    def find_GO_name_from_number(self, go_number):
        for key, value in self.data.items():
            if value['exactSource'] == go_number:
                return key
        return None

    def get_num_terms(self):
        return len(self.data)

    def get_genes_for_term(self, term_name):
        """
        Return the list of gene symbols (value['geneSymbols']) for a given MSigDB
        term name. E.g. "GOBP_THYMOCYTE_MIGRATION".
        Returns None if the term_name is not found.
        """
        if term_name in self.data:
            return self.data[term_name].get('geneSymbols', [])
        return None

    def __iter__(self):
        self.iter_index = iter(self.data.items())
        return self

    def __next__(self):
        key, value = next(self.iter_index)
        go_term_number = value['exactSource']
        go_name = key
        genes = value['geneSymbols']
        return go_term_number, go_name, genes

