import STCase
import argparse


parser = argparse.ArgumentParser(description="STCase for CCI sub-clustering on Spatial-Omics") # Tid: HACK
parser.add_argument('--root', required=True, help="which root to do CCI prediction", type=str)
parser.add_argument('--ds-dir', required=True, type=str, metavar='DATA', help='dataset directory')
parser.add_argument('--ds-name', required=True, type=str, metavar='DATA', help='dataset name')
parser.add_argument('--h5-name', required=True, type=str, metavar='DATA', help='h5ad file name')
parser.add_argument('--label-col-name', required=True, type=str, metavar='DATA', help='label column name')
parser.add_argument('--target-types', nargs='*', default=[], type=str, help="List of target cell-types", required=True)
parser.add_argument('--bad-types', nargs='*', default=[], type=str, help="List of bad cell-types")
parser.add_argument('--n-nei', type=int, required=True)
parser.add_argument('--n-clusters', type=int, default=-1)
parser.add_argument('--alpha', type=float, default=0.0)
parser.add_argument('--reso', type=float, default=None)
parser.add_argument('--init', default='one', type=str, help="init weight", choices=['one', 'std', 're_sum', 'sum'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use-gpu', help="use gpu acceleration", action='store_true')
parser.add_argument('--wo-anno', help="without groundtruth annotation for sub-clustering", action='store_true')
parser.add_argument('--region-col-name', default="NULL", type=str, metavar='DATA', help='Sub-clustering region ground-truth label column name')
test_args = parser.parse_args()


if __name__ == '__main__':
    # Test
    stcase_args = STCase.prepare(test_args)
    STCase.train(stcase_args)
