#!/usr/bin/env python3
import pickle
import argparse
from neural_network import prepare_database, triplet_loss, who_is_it, DATABASE_PATH

create_notification = False
try:   # only create notification if module present
    from plyer import notification
    create_notification = True
except ModuleNotFoundError:
    pass

"""
Scan the images directory for folders represnting identities.
For each identity folder found load images within the folder and get their encodings (1x128 tensors produced from the inception model)
This creates and in-memory database like: {"name": [tensor1...tensor_n], ...}
The dictionary is then saved as a pickle file to the database directory
Usage: $>python3 refresh_database.py [--use_avg]
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update the database to reflect the current structure in the project's images directory")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--use_avg', dest='feature', action='store_true')
    # feature_parser.add_argument('--use_sum', dest='feature', action='store_false')
    parser.set_defaults(feature=False)
    args = parser.parse_args()

    database = prepare_database(use_avg=args.feature)
    with open(DATABASE_PATH, "wb") as f:
        pickle.dump(database, f)
    if create_notification:
        notification.notify(
            title="Database updated",
            message=f"Facial recognition database updated with {len(database)} {'entry' if len(database) == 1 else 'entries'}",
            app_name="Facial Recognition",
            # app_icon='path/to/the/icon.png'
        )
