import psana
from datetime import datetime
begin_job_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
import time
start_job = time.time()
import numpy as np
import argparse
import socket
import os
import logging
import requests
import sys
from glob import glob
from requests.auth import HTTPBasicAuth
from pathlib import Path
from typing import Optional, Tuple
#import panel as pn
#import holoviews as hv
#from holoviews import dim
#hv.extension('bokeh')
#pn.extension()

def getDetParams(run, detnames):
    """Set parameters for checking detector events."""
    ret_dict = {}
    for detname in detnames.split(','):
        if int(run)>0:
            if detname == 'Rayonix':
                ret_dict['Rayonix'] = {'det_shape':(2560, 2560)}
            else:
                raise NotImplementedError(f'unrecognized detector address: {detname}')
    return ret_dict

def getEbeam(evt):
    source = psana.Source('BldInfo(EBeam)')
    ebeam = evt.get(psana.Bld.BldDataEBeamV6, source) or \
            evt.get(psana.Bld.BldDataEBeamV5, source) or \
            evt.get(psana.Bld.BldDataEBeamV4, source) or \
            evt.get(psana.Bld.BldDataEBeamV3, source) or \
            evt.get(psana.Bld.BldDataEBeamV2, source) or \
            evt.get(psana.Bld.BldDataEBeamV1, source) or \
            evt.get(psana.Bld.BldDataEBeamV0, source) or \
            evt.get(psana.Bld.BldDataEBeam, source)
    return ebeam

def getFEE(evt, ds, evt_idx):
    det = psana.Detector('FEE-SPEC0')
    fee = det.get(evt)
    return fee

def isDropped(def_data):
    if def_data['lightStatus']['xray'] == 0: 
        return True
    return False

def defineDets(run, detname):
    try:
        detParams = getDetParams(run, detname)
    except Exception as e:
        print(f'Can\'t instantiate args: {e}')
        sys.exit()
    dets = {}

    # Define detectors and their associated DetObjectFuncs
    for detname in detParams:
        havedet = checkDet(ds.env(), detname)
        det = psana.Detector(detname)
        det._expected_shape = detParams[detname]['det_shape']
        dets[detname] = det
    return dets

##########################################################
# Custom exception handler to make job abort if a single rank fails.
# Avoid jobs hanging forever and report actual error message to the log file.
import traceback as tb

def global_except_hook(exctype, value, exc_traceback):
    tb.print_exception(exctype, value, exc_traceback)
    sys.stderr.write("except_hook. Calling MPI_Abort().\n")
    sys.stdout.flush() # Command to flush the output - stdout
    sys.stderr.flush() # Command to flush the output - stderr
    # Note: mpi4py must be imported inside exception handler, not globally.     
    import mpi4py.MPI
    mpi4py.MPI.COMM_WORLD.Abort(1)
    sys.__excepthook__(exctype, value, exc_traceback)
    return

sys.excepthook = global_except_hook
##########################################################


# General Workflow
# This is meant for arp which means we will always have an exp and run
# Check if this is a current experiment
# If it is current, check in ffb for xtc data, if not there, default to psdm

fpath=os.path.dirname(os.path.abspath(__file__))
fpathup = '/'.join(fpath.split('/')[:-1])
sys.path.append(fpathup)
print(f'\nadding {fpathup} to $PATH')

from smalldata_tools.utilities import checkDet
from smalldata_tools.SmallDataUtils import defaultDetectors, detData
#from smalldata_tools.DetObject import DetObject
#from summaries.BeamlineSummaryPlots_mfx import postElogMsg

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
#logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Constants
HUTCHES = [
	'AMO',
	'SXR',
	'XPP',
	'XCS',
	'MFX',
	'CXI',
	'MEC',
	'DIA'
]

S3DF_BASE = Path('/sdf/data/lcls/ds/')
FFB_BASE = Path('/cds/data/drpsrcf/')
PSANA_BASE = Path('/cds/data/psdm/')
PSDM_BASE = Path(os.environ.get('SIT_PSDM_DATA', S3DF_BASE))
EVT_EXT = Path('./results/common/event_monitor/')
logger.debug(f"PSDM_BASE={PSDM_BASE}")

# Define Args
parser = argparse.ArgumentParser()
parser.add_argument('--run', 
                    help='run', 
                    type=str, 
                    default=os.environ.get('RUN_NUM', ''))
parser.add_argument('--experiment', 
                    help='experiment name', 
                    type=str, 
                    default=os.environ.get('EXPERIMENT', ''))
parser.add_argument('--stn', 
                    help='hutch station', 
                    type=int, 
                    default=0)
parser.add_argument('--nevents', 
                    help='number of events', 
                    type=int, 
                    default=1e9)
parser.add_argument('--detectors',
                    help='comma separated addresses of detectors to monitor',
                    type=str,
                    default=None)
parser.add_argument('--directory',
                    help='directory for output files (def <exp>/event_monitor)')
parser.add_argument('--gather_interval', 
                    help='gather interval', 
                    type=int, 
                    default=100)
parser.add_argument('--url',
                    default="https://pswww.slac.stanford.edu/ws-auth/lgbk/")
parser.add_argument("--wait", 
                    help="wait for a file to appear",
                    action='store_true', 
                    default=False)
parser.add_argument("--postElog",
                    help="post alerts to eLog",
                    action='store_true',
                    default=False)
args = parser.parse_args()
logger.debug('Args to be used for event monitoring: {0}'.format(args))

###### Helper Functions ##########

def get_xtc_files(base, exp, run):
    """File all xtc files for given experiment and run"""
    run_format = ''.join(['r', run.zfill(4)])
    data_dir = Path(base) / exp[:3] / exp / 'xtc'
    xtc_files = list(data_dir.glob(f'*{run_format}*'))
    logger.info(f'xtc file list: {xtc_files}')
    return xtc_files

def get_log_file(write_dir, exp, hutch):
    """Generate directory to write to, create file name"""
    if write_dir is None:
        if useFFB and not onS3DF: # when on a drp node
            write_dir = FFB_BASE / hutch.lower() / exp / '/scratch' / EVT_EXT
        elif onPSANA: # when on old psana system
            write_dir = PSANA_BASE / hutch.lower() / exp, EVT_EXT
        elif onS3DF: # S3DF should now be the default
            write_dir = S3DF_BASE / hutch.lower() / exp / EVT_EXT
        else:
            print('get_log_file problem. Please fix.')
    logger.debug(f'event monitor log directory: {write_dir}')

    write_dir = Path(write_dir)
    log_fname = write_dir / f'{exp}_run{run.zfill(4)}_evts.log'
    if not write_dir.exists():
        logger.info(f'{write_dir} does not exist, creating directory now.')
        try:
            write_dir.mkdir(parents=True)
        except (PermissionError, FileNotFoundError) as e:
            logger.info(f'Unable to make directory {write_dir} for output' \
                        f'exiting on error: {e}')
            sys.exit()
    logger.info('Will write event monitoring log to {0}'.format(log_fname))
    return log_fname

def getElogBasicAuth(exp: str) -> HTTPBasicAuth:
    """Return an authentication object for the eLog API for an opr account.

    This method will only work for active experiments. "opr" accounts are
    removed from the authorized users list after the experiment ends.

    Paramters
    ---------
    exp (str) Experiment name (to determine operator username).

    Returns
    -------
    http_auth (HTTPBasicAuth) Authentication for eLog API.
    """
    opr_name: str = f"{exp[:3]}opr"
    hostname: str = socket.gethostname()
    if hostname.find('sdf') >= 0:
        auth_path: str = "/sdf/group/lcls/ds/tools/forElogPost.txt"
    else:
        auth_path: str = f"/cds/home/opr/{opr_name}/forElogPost.txt"

    with open(auth_path, "r") as f:
        pw: str = f.readline()[:-1]

    return HTTPBasicAuth(username=opr_name, password=pw)

def postElogMsg(
        exp: str,
        msg: str,
        *,
        tag: Optional[str] = "",
        title: Optional[str] = "",
        files: list = []
) -> None:
    """Post a new message to the eLog. Adapted from `elog` package.

    Parameters
    ----------
    exp (str) Experiment name.
    msg (str) Body of the eLog post.
    tag (str) Optional. A tag to include for the post.
    title (str) Optional. A title for the eLog post.
    files (list) Optional. Either a list of paths (str) to files (figures) to
        include with the eLog post, OR, a list of 2-tuples of strings of the
        form (`path`, `description`).
    """
    post_files: list = []
    for f in files:
        if isinstance(f, str):
            desc: str = os.path.basename(f)
            formatted_file: tuple = (
                "files",
                (desc, open(f, "rb")),
                mimetypes.guess_type(f)[0]
            )
        elif isinstance(f, tuple) or isinstance(f, list):
            formatted_file: tuple = (
                "files",
                (f[1], open(f[0], "rb")),
                mimetypes.guess_type(f[0])[0]
            )
        else:
            logger.debug(f"Can't parse file {f} for eLog attachment. Skipping.")
            continue
        post_files.append(formatted_file)

    post: dict = {}
    post['log_text'] = msg
    if tag:
        post['log_tags'] = tag
    if title:
        post['log_title'] = title

    http_auth: HTTPBasicAuth = getElogBasicAuth(exp)
    base_url: str = "https://pswww.slac.stanford.edu/ws-auth/lgbk/lgbk"
    post_url: str = f"{base_url}/{exp}/ws/new_elog_entry"

    params: dict = {'url': post_url, 'data': post, 'auth': http_auth}
    #params: dict = {'url': post_url, 'data': post}
    if post_files:
        params.update({'files': post_files})

    resp: requests.models.Response = requests.post(**params)

    if resp.status_code >= 300:
        logger.debug(
            f"Error when posting to eLog: HTTP status code {resp.status_code}"
        )

    if not resp.json()['success']:
        logger.debug(f"Error when posting to eLog: {resp.json()['error_msg']}")



def postMissingDataMsg(
        data_items: dict,
        exp: str,
        run: int,
        *,
        tag: str = "SUMMARY_DATA_MISSING",
        title: str = "MISSING DATA ALERT",
        post_thresh: float = 0.1
) -> None:
    """Post alarm to the eLog indicating a specified run is missing one or more data types.

    Parameters
    ----------
    data_items (dict{str:(int, int, float)}) Dictionary of data items to report on. Keys are identifiers in psana events and values are tuples of (1) the counts of missing or damaged events, (2) the total event count for the run, and (3) the bad event rate.
    exp (str) Experiment name.
    run (int) Run number. Usually the current run.
    tag (str) Optional. Tag for the event damage summary posts.
    title (str) Optional. Title for event damage summary posts.
    post_thresh (float) Optional. Missing data threshold (as a percentage)
        required to post a message to eLog.
    """
    table_header: str = (
        '<thead><tr><th colspan="3">'
        f'<center>Summary of missing data in run {run}</center>'
        '</th></tr></thead>'
    )
    table_body: str = (
        '<tbody><tr>'
        '<td><b><center>Data Item</center></b></td>'
        '<td><b><center>Event Count</center></b></td>'
        '<td><b><center>Missing Data Rate</center></b></td></tr>'
    )
    
    post_msg: bool = False

    for name in data_items:
        missing_count, event_count, missing_pct = data_items[name]
        if missing_pct >= post_thresh:
            post_msg = True
        entry: str = (
            f'<tr><td><center>{name}</center></td>'
            f'<td><center>{int(missing_count)} of {event_count} events missing data</center></td>'
            f'<td><center>{missing_pct:.2%}</center></td></tr>'
        )
        table_body += entry
    table_body += '</tbody>'
    msg: str = f'<table border="1">{table_header}{table_body}</table>'
    if post_msg:
        postElogMsg(exp=exp, msg=msg, tag=tag, title=title)
        # missing post request -- see request package (as user, not operator)
        print('Posted alert to the eLog.')

##### START SCRIPT ########

# Define hostname
hostname = socket.gethostname()

# Parse hutch name from experiment and check it's a valid hutch
exp = args.experiment
run = args.run
station = args.stn
detnames = args.detectors
logger.debug(f'Checking event data for EXP:{exp} - RUN:{run}')
logger.debug(f'Looking for detectors {detname}')

begin_prod_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')

hutch = exp[:3].upper()
if hutch not in HUTCHES:
	logger.debug('Could not find {0} in list of available hutches'.format(hutch))
	sys.exit()

# Figure out where we are and where to look for data
xtc_files = []
useFFB = False
onS3DF = False
onPSANA = False

if hostname.find('sdf')>=0:
    logger.debug('On S3DF')
    onS3DF = True
    if 'ffb' in PSDM_BASE.as_posix():
        useFFB = True
        # wait for files to appear
        nFiles = 0
        n_wait = 0
        max_wait = 20 # 10s wait per cycle.
        waitFilesStart=datetime.now()
        while nFiles == 0:
            if n_wait > max_wait:
                print(f"Waited {str(n_wait*10)}s, still no files available." \
                       "Giving up, please check dss nodes and data movers." \
                       "Exiting now.")
                sys.exit()
            xtc_files = get_xtc_files(PSDM_BASE, exp, run)
            nFiles = len(xtc_files)
            if nFiles == 0:
                print(f"We have no xtc files for run {run} in {exp} in the FFB system," \
                      "we will wait for 10 second and check again.")
                n_wait+=1
                time.sleep(10)
        waitFilesEnd = datetime.now()
        print(f"Files appeared after {str(waitFilesEnd-waitFilesStart)} seconds")

    xtc_files = get_xtc_files(PSDM_BASE, exp, run)
    if len(xtc_files)==0:
        print(f'We have no xtc files for run {run} in {exp} in the offline system. Exit now.')
        sys.exit()

elif hostname.find('drp')>=0:
    nFiles=0
    logger.debug('On FFB')
    waitFilesStart=datetime.now()
    while nFiles==0:
        xtc_files = get_xtc_files(FFB_BASE, hutch, run)
        nFiles = len(xtc_files)
        if nFiles == 0:
            if not args.wait:
                print("We have no xtc files for run %s in %s in the FFB system,"\
                      "Quitting now."%(run,exp))
                sys.exit()
            else:
                print("We have no xtc files for run %s in %s in the FFB system," \
                      "we will wait for 10 second and check again."%(run,exp))
                time.sleep(10)
    waitFilesEnd = datetime.now()
    print('Files appeared after %s seconds'%(str(waitFilesEnd-waitFilesStart)))
    useFFB = True

# If not a current experiment or files in ffb, look in psdm
else:
    logger.debug('Not on FFB or S3DF, use old offline system')
    xtc_files = get_xtc_files(PSDM_BASE, hutch, run)
    if len(xtc_files)==0:
        print('We have no xtc files for run %s in %s in the offline system'%(run,exp))
        sys.exit()

# Get output file, check if we can write to it
log_fname = get_log_file(args.directory, exp, hutch)

# Define data source and instantiate data source object
ds_name = f'exp={exp}:run={run}'
logger.debug(f'DataSource name: {ds_name}')

# # note, MPIDataSource functions very differently from DataSource
#try:
#    ds_mpi = psana.MPIDataSource(ds_name)
#except Exception as e:
#    logger.debug('Could not instantiate MPIDataSource with {0}: {1}'.format(ds_name, e))
#    sys.exit()

try:
    ds = psana.DataSource(f'{ds_name}:idx')
except Exception as e:
    logger.debug('Could not instantiate DataSource with {0}: {1}'.format(ds_name, e))
    sys.exit()

# Set up detector objects
start_setup_dets = time.time()
default_dets = defaultDetectors(hutch.lower())
default_det_aliases = [det.name for det in default_dets]
dets = defineDets(run, detname)
end_setup_dets = time.time()

start_evt_loop = time.time()
run_ds = next(ds.runs())
times = run_ds.times()
events_count = min([args.nevents, len(times)])
event_status = {name:np.ones(events_count) for name in dets.keys()}
event_status.update({'FEE':np.ones(events_count),
                     'ebeam':np.ones(events_count),
                     'xrays': np.ones(events_count)})

for i in range(events_count):
    evt = run_ds.event(times[i])
    def_data = detData(default_dets, evt)

    # check detectors
    dets_statuses = []
    if isDropped(def_data):
        print(f'Event {i} dropped.')
        for datatype, array in event_status.items():
            array[i] = 0
        continue
    for detname, det in dets.items():
        try:
            data = det.image(evt)
            if data.shape == det._expected_shape:
                status = 'ok'
            else:
                status = f'damaged: data shape is {data.shape}'
                event_status[detname][i] = 0
        except Exception:
            status = 'missing'
            event_status[detname][i] = 0
        dets_statuses.append(f'{detname}: {status}')
    dets_statuses_str = ', '.join(dets_statuses)

    # check for ebeam and FEE
    try:
        ebeam = getEbeam(evt)
        assert ebeam is not None
        ebeam_status = 'ok'
    except Exception:
        ebeam_status = 'missing'
        event_status['ebeam'][i] = 0
    try:
        fee = getFEE(evt, ds, i)
        assert fee is not None
        fee_status = 'ok'
    except Exception:
        fee_status = 'missing'
        event_status['FEE'][i] = 0
    print(f'Event {i}: FEE is {fee_status}, ebeam is {ebeam_status}, detector(s) are {dets_statuses_str}')

    #the ARP will pass run & exp via the enviroment, if I see that info, the post updates
    if ( (i<100 and i%10==0) or (i<1000 and i%100==0) or (i%1000==0)):
        if os.environ.get('ARP_JOB_ID', None) is not None:
            requests.post(os.environ["JID_UPDATE_COUNTERS"], json=[{"key": "<b>Current Event</b>", "value": i+1}])
        else:
            print('Current Event:', i+1)

# print summary
print(f'\nRun {run} summary:')
shots_count = len(event_status['xrays'])
events_count = int(sum(event_status['xrays'])) # exclude dropped shots from "events"
shots_dropped = shots_count - events_count
print(f'{shots_dropped} shots dropped with no xray data.')
summaries = {}
all_ok = np.ones(shots_count)
for name, oks in event_status.items():
    all_ok = all_ok * oks
    if name == 'xrays':
        summaries[name] = (shots_dropped, shots_count, shots_dropped/shots_count)
    not_ok_count = events_count - int(sum(oks))
    if not_ok_count:
        print(f'{int(not_ok_count)} events have missing or damaged {name}.')
    bad_rate = not_ok_count / events_count
    summaries[name] = (not_ok_count, events_count, bad_rate)
all_ok_count = sum(all_ok)
print(f'{int(all_ok_count)} of {events_count} events with xrays in run {run} are OK. {int(events_count - all_ok_count)} are damaged or missing.')

if args.postElog:
    # post alert to main eLog if >10% of events are missing for any data item
    postMissingDataMsg(summaries, exp, run)

end_evt_loop = time.time()

# Print duration summary
dets_time_start = (start_setup_dets-start_job)/60
dets_time_end = (end_setup_dets-start_job)/60
evt_time_start = (start_evt_loop-start_job)/60
evt_time_end = (end_evt_loop-start_job)/60
logger.debug(f"##### Timing benchmarks core: ##### ")
logger.debug(f'Setup dets: \n\tStart: {dets_time_start:.2f} min\n\tEnd: {dets_time_end:.2f} min')
logger.debug(f'\tDuration:{dets_time_end-dets_time_start:.2f}')
logger.debug(f'Event loop: \n\tStart: {evt_time_start:.2f} min\n\tEnd: {evt_time_end:.2f} min')
logger.debug(f'\tDuration:{evt_time_end-evt_time_start:.2f}')
logger.debug('\n')

end_prod_time = datetime.now().strftime('%m/%d/%Y %H:%M:%S')
end_job = time.time()
prod_time = (end_job-start_job)/60
#print('########## JOB TIME: {:03f} minutes ###########'.format(prod_time))

