import py3Dmol as pm
from mlatom import molecule

styles: [str] = [
    'line',
    'stick',
    'sphere',
    'cartoon',
    'ballstick',
    'cross',
    'surface',
    'ribbon',
    'points'
]

is_ipykernel = False

def visualize_molecule(mol: molecule):
    if is_ipykernel:
        mol.view()

def visualize_two_molecules(xyz_data1: str, xyz_data2: str, width: int = 800, height: int = 600):
    if is_ipykernel:
        view = pm.view(width=width, height=height)
        view.addModel(xyz_data1, "xyz")
        view.setStyle({"model": 0}, {"stick": {'color': 'yellow'}, "sphere": {'color': 'yellow', 'scale': 0.25}})
        view.addModel(xyz_data2, "xyz")
        view.setStyle({"model": 1}, {"stick": {'color': 'blue'}, "sphere": {'color': 'blue', 'scale': 0.25}})
        view.zoomTo()
        view.show()
    else:
        print('*** Unable to visualize without ipykernel ***')

def check_is_ipykernel() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if 'ZMQInteractiveShell' in shell:
            ipykernel = True
        else:
            ipykernel = False
    except:
        ipykernel = False
    if not ipykernel:
        print('Ipykernel not available - will skip visualizations.')
    return ipykernel

