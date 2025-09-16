import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_config', required=True, type=str,
                        help='Path to JSON config file for the dataset')
    parser.add_argument('--adios_in', required=True, type=str,
                        help='Path to ADIOS input file')
    parser.add_argument('--adios_out', required=True, type=str,
                        help='Path to ADIOS output file')
    parser.add_argument('--use_intermediate_db', action='store_true',
                        help='Set to true if you want to store partially processed data in a database and resume '
                             'processing in a subsequent batch job.')
    parser.add_argument('--db_path', default="./db/", type=str,
                        help='Path to intermediate directory containing db files')

    args = parser.parse_args()
    return args
