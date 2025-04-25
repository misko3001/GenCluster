import ast
from typing import Optional, List


def get_class_names(file_path: str, exclude: Optional[List[str]]) -> List[str]:
    if exclude is None:
        exclude = []
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read())
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name not in exclude]