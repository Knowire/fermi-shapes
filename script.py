#!/usr/bin/env python3
from argparse import ArgumentParser
from configparser import ConfigParser
from xml.etree import ElementTree as ET
from os import path

DEFAULT_BINSIZE = 0.001 # MeV

# parse args and config
parser = ArgumentParser()
group1, group2 = parser.add_mutually_exclusive_group(), parser.add_mutually_exclusive_group() 
parser.add_argument('filepath', help='path to decay file, or config if --config-mode')
parser.add_argument('-c', '--config-mode', action='store_true', help='if specified, then filepath leads to config')
parser.add_argument('--all-allowed', action='store_true', help='if chosen, every transition is treated as allowed')
parser.add_argument('-r', '--use-ref', action='store_true', help='if chosen, use reference files')
group1.add_argument('--binsize', type=float, help=f'bin size in MeV (float), default is {DEFAULT_BINSIZE}')
group1.add_argument('--datalen', type=int, help='number of bins (int)')
group2.add_argument('-p', '--plot', action='store_true', help='whether to plot result at the end of run')
group2.add_argument('-o', '--output', help='output file path')
args = parser.parse_args()

if not path.exists(args.filepath):
    parser.error('file not found')

config = ConfigParser()
if args.config_mode:
    config.read(args.filepath)
    DECAY_PATH = config.get('Basic', 'decay_file_path')
    if not path.exists(DECAY_PATH):
        parser.error('config: decay file not found')
else:
    DECAY_PATH = args.filepath

if path.splitext(DECAY_PATH)[-1].lower() != '.xml':
    parser.error('wrong decay file extension - should be xml')

OUTPUT_PATH = args.output or config.get('Basic', 'output_file_path', fallback='out.csv')
PLOT = args.plot or config.getboolean('Basic', 'plot', fallback=False)
USE_FORBIDDEN = False if args.all_allowed else config.getboolean('Basic', 'use_forbidden', fallback=True)
REF = args.use_ref or config.getboolean('Basic', 'use_reference_files', fallback=False)

if args.datalen!=None or args.binsize!=None:
    DATALEN = args.datalen
    BINSIZE = args.binsize
else:
    DATALEN = config.getint('Basic', 'datalen', fallback=None)
    BINSIZE = config.getfloat('Basic', 'binsize', fallback=None)

if DATALEN != None and DATALEN <= 0: parser.error('datalen must be greater than 0')
elif BINSIZE != None and BINSIZE <= 0: parser.error('binsize must be greater than 0')

# read XML
with open(DECAY_PATH) as f:
    next(f)
    decay = ET.fromstringlist(["<root>", f.read(), "</root>"])

nuclide_files = decay.findall('NuclideFile')
directory = path.dirname(DECAY_PATH)

start_level = decay.find('StartLevel')
start_Z, start_A = start_level.attrib['AtomicNumber'], start_level.attrib['AtomicMass']

nuclide1, nuclide2 = None, None
for nf in nuclide_files:
    nf_path = path.join(directory, nf.attrib['FileName'])
    nuclide = ET.parse( nf_path ).getroot()
    nuclide_Z, nuclide_A = nuclide.attrib['AtomicNumber'], nuclide.attrib['AtomicMass']
    if (start_Z, start_A) == (nuclide_Z, nuclide_A): nuclide_file1, nuclide_file_path1, nuclide1 = nf, nf_path, nuclide; break
if nuclide1 is None:
    parser.error('parent/mother (StartLevel) nuclide file not found')

BETA_PLUS = nuclide1.find('Level').find('Transition').attrib['Type'] != 'B-'
stop_Z, stop_A = str(int(start_Z)-1) if BETA_PLUS else str(int(start_Z)+1), start_A
nuclide_files.remove(nuclide_file1)

for nf in nuclide_files:
    nf_path = path.join(directory, nf.attrib['FileName'])
    nuclide = ET.parse( nf_path ).getroot()
    nuclide_Z, nuclide_A = nuclide.attrib['AtomicNumber'], nuclide.attrib['AtomicMass']
    if (stop_Z, stop_A) == (nuclide_Z, nuclide_A): nuclide_file2, nuclide_file_path2, nuclide2 = nf, nf_path, nuclide; break
if nuclide2 is None:
    parser.error('child/daughter nuclide file not found')

paths = [DECAY_PATH, nuclide_file_path1, nuclide_file_path2]

if REF:
    try:
        ref_nuclide_file_path1 = path.join(directory, nuclide_file1.attrib['RefFileName'])
        ref_nuclide_file_path2 = path.join(directory, nuclide_file2.attrib['RefFileName'])
        ref_nuclide1, ref_nuclide2 = ET.parse( ref_nuclide_file_path1 ).getroot(), ET.parse( ref_nuclide_file_path2 ).getroot()
        paths += [ref_nuclide_file_path1, ref_nuclide_file_path2]
    except:
        parser.error('RefFileName attributes not set, or specified files not found')

# analyse ==============================================================================
import numpy as np
from fermi import FermiBeta
from utils import neutrino_cs

Z = float(nuclide1.attrib['AtomicNumber'])
A = float(nuclide1.attrib['AtomicMass'])
T_max = float(nuclide1.attrib['QBeta'])/1000 # MeV

beta_decay = FermiBeta(Z, A, BETA_PLUS)
T = np.linspace(0, T_max, DATALEN) if DATALEN else np.arange(0, T_max, BINSIZE or DEFAULT_BINSIZE)

def get_transition_type(sp1, sp2, par1, par2):
    if sp1 and sp2:
        sp1, sp2 = float(sp1), float(sp2)
        if sp1>=0 and sp2>=0:
            is_parity_changed = par1!=par2
            spin_delta = int(np.abs( sp1-sp2 ))
            return (spin_delta, is_parity_changed)
    else:
        return None

def analyse(nuc1, nuc2):
    P_e, P_nu = np.zeros(len(T)), np.zeros(len(T))

    start_level = nuc1.find('Level')
    spin1, parity1 = start_level.attrib['Spin'], start_level.attrib['Parity']
    transitions = start_level.findall('Transition')

    if USE_FORBIDDEN:
        for t in transitions:
            target_level_energy = t.find('TargetLevel').attrib['Energy']
            target_level = nuc2.find(f'Level[@Energy="{target_level_energy}"]')
            spin2, parity2 = target_level.attrib['Spin'], target_level.attrib['Parity']
            transition_type = get_transition_type(spin1, spin2, parity1, parity2)

            P_e += beta_decay.electron_shape(
                T, float(t.attrib['TransitionQValue'])/1000, float(t.attrib['Intensity']), transition_type)  
            P_nu += beta_decay.neutrino_shape(
                T, float(t.attrib['TransitionQValue'])/1000, float(t.attrib['Intensity']), transition_type)  
    else:
        for t in transitions:
            P_e += beta_decay.electron_shape(T, float(t.attrib['TransitionQValue'])/1000, float(t.attrib['Intensity']))
            P_nu += beta_decay.neutrino_shape(T, float(t.attrib['TransitionQValue'])/1000, float(t.attrib['Intensity']))

    return P_e, P_nu


columns = {}
columns['P_e'], columns['P_nu'] = analyse(nuclide1, nuclide2)
if not BETA_PLUS:
    columns['cs_P_nu'] = neutrino_cs(T)*columns['P_nu']
if REF:
    columns['ref_P_e'], columns['ref_P_nu'] = analyse(ref_nuclide1, ref_nuclide2)
    if not BETA_PLUS:
        columns['ref_cs_P_nu'] = neutrino_cs(T)*columns['ref_P_nu']

if PLOT:
    import matplotlib.pyplot as plt
    from figures import make_ref_fig, make_fig
    fig = make_ref_fig(T, *columns.values(), file_paths=paths) if REF else make_fig(T, *columns.values(), file_paths=paths)
    plt.show()
else:
    import csv
    with open(OUTPUT_PATH, 'w', newline='') as csvfile:
        rows = zip(T, *columns.values())
        fieldnames = ['T', *columns.keys()]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row_tuple in rows:
            writer.writerow(dict(zip(fieldnames, row_tuple)))
