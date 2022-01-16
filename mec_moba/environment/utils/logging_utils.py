import csv
import os.path
import typing
from collections.abc import Iterable

_opened_log_files = {}


def close_all_logs():
    for _, (csv_writer, csv_file) in _opened_log_files.items():
        csv_file.close()


def log_to_csv_file(base_log_dir,
                    log_filename,
                    data_to_log: typing.Union[typing.NamedTuple, typing.Iterable[typing.NamedTuple]]):
    if not isinstance(data_to_log, Iterable):
        data_to_log = [data_to_log]

    if len(data_to_log) > 0:
        log_file_key = (base_log_dir, log_filename)
        if log_file_key not in _opened_log_files:
            os.makedirs(base_log_dir, exist_ok=True)
            csv_file = open(os.path.join(base_log_dir, log_filename), 'w', newline='')
            csv_writer = csv.writer(csv_file, delimiter=',')
            # write header row
            csv_writer.writerow(data_to_log[0]._fields)
            _opened_log_files[log_file_key] = (csv_writer, csv_file)

        csv_writer, csv_file = _opened_log_files[log_file_key]

        csv_writer.writerows(data_to_log)
