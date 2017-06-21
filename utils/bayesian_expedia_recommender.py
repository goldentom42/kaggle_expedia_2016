"""
Validation produces the following results

               HITS@1  %USE  %ACC    HITS@2  %USE  %ACC    HITS@3  %USE  %ACC    HITS@4  %USE  %ACC    HITS@5  %USE  %ACC
LEAK          0.25021 29.24 85.58 | 0.01822  5.86 31.11 | 0.00115  0.67 17.06 | 0.00005  0.05 10.29 | 0.00000  0.00  3.51 |
UID_DEST      0.02443  9.62 25.38 | 0.00800 10.32  7.75 | 0.00365  7.86  4.64 | 0.00205  5.59  3.67 | 0.00126  4.02  3.14 |
DEST_MKT_PACK 0.11119 60.56 18.36 | 0.07837 82.44  9.51 | 0.06073 89.14  6.81 | 0.05004 91.04  5.50 | 0.04080 91.60  4.45 |
DEST          0.00019  0.07 27.93 | 0.00017  0.19  9.07 | 0.00014  0.30  4.85 | 0.00014  0.38  3.74 | 0.00014  0.51  2.75 |
HOT_CO_MKT    0.00053  0.51 10.25 | 0.00051  1.19  4.26 | 0.00055  2.03  2.73 | 0.00062  2.92  2.14 | 0.00061  3.83  1.59 |
HOT_CO        0.00000  0.00  0.00 | 0.00000  0.00  0.00 | 0.00000  0.00  1.69 | 0.00000  0.01  1.10 | 0.00000  0.03  0.00 |
Total MAP =  0.48304672758905076

5 minutes to build the recommendation system
4 minutes to apply it on the validation set
"""

import pandas as pd
import numpy as np
from heapq import nlargest
from operator import itemgetter
from collections import defaultdict
import time


class BayesianExpediaReco(object):
    def __init__(self,
                 nb_recos=5,
                 do_submission=False,
                 weight_type=None,
                 name='standard'):
        self.nb_recos = nb_recos
        self.best_hotels = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.validation_stats = defaultdict(lambda: {x: [0, 0] for x in range(self.nb_recos)})
        self.the_keys = None
        self.do_submission = do_submission
        self.validation_recos = 0
        self.name = name
        self.weight_strategy = weight_type

    def register_key(self, the_name=None, the_key=None, the_weight=None):
        if self.the_keys is None:
            self.the_keys = [(the_name, the_key, the_weight)]
        else:
            self.the_keys.append((the_name, the_key, the_weight))

    def update_reco(self, row):
        """
        This version takes 16 minutes with keys (hotel_country)  to compute best_hotels
        and produces
        Results for keys ['hotel_country']
        % Hits @ place 0 : 0.08245
        % Hits @ place 1 : 0.05837
        % Hits @ place 2 : 0.04207
        % Hits @ place 3 : 0.03628
        % Hits @ place 4 : 0.03226
        0.14118029375473137
        """
        # Build dict key
        # The list comprehension is a lot quicker than a simple dict_key = tuple(row[self.the_keys])
        # It's almost 5 times quicker !!
        # This version takes with keys (user_location_city, orig_destination_distance)
        dict_key = tuple([row[key] for key in self.the_keys])

        # Update best_hotels dict
        # Note that all values in the keys are input because of the dropna in build_reco
        self.best_hotels[dict_key][row['hotel_cluster']] += row['weight']

    def update_hotels(self, key_nb, dict_key, row):
        self.best_hotels[key_nb][dict_key][row[-1]] += row[-2]

    def update_reco_raw(self, row, len_keys=None, key_nb=None):
        """
        The raw version takes the row argument as an ndarray
        the first items are the key and the last 2 items are the weight and the hotel_cluster
        This version takes 1.4 minutes with keys (hotel_country) to compute best_hotels
        and produces
        Results for keys ['hotel_country']
        % Hits @ place 0 : 0.08245
        % Hits @ place 1 : 0.05837
        % Hits @ place 2 : 0.04207
        % Hits @ place 3 : 0.03628
        % Hits @ place 4 : 0.03226
        0.14118029375473137
        """
        # Build dict key
        # The list comprehension is a lot quicker than a simple dict_key = tuple(row[self.the_keys])
        # It's almost 5 times quicker !!
        # This version takes
        dict_key = tuple(row[:len_keys])

        # Update best_hotels dict
        # Note that all values in the keys are input because of the dropna in build_reco
        self.update_hotels(key_nb, dict_key, row)

    @staticmethod
    def update_weights(df, ref_type='event'):
        # Get reference date_time for weight computation
        if ref_type == 'event':
            ref = 'date_time'
        elif ref_type == 'checkin':
            ref = 'srch_ci'
        elif ref_type == 'checkout':
            ref = 'srch_co'
        else:
            raise ValueError('Unknown reference for weights. Ref should be one of event, checkin, checkout')

        # treat Null dates and fallback to event dates
        df.loc[df[ref].isnull(), ref] = df.loc[df[ref].isnull(), 'date_time']
        # Get month and year
        df['year'] = df[ref].str.split('-').str[0].astype(int)
        df['month'] = df[ref].str.split('-').str[1].astype(int)

        # Compute weights
        df['weight0'] = ((df['year'] - 2012) * 12 + (df['month'] - 12))
        df.loc[df['weight0'] < 0, 'weight0'] = 0  # i.e. year < 2013
        df['weight1'] = np.power(df['weight0'], 1.5) * (3 + 17.6 * df['is_booking'])
        df['weight2'] = 3 * np.floor((df['month'] + 1) / 4) + 5.56 * df['is_booking']

        # Make sure dates < 2013 are zero weighted
        df.loc[df['year'] < 2013, 'weight0'] = 0
        df.loc[df['year'] < 2013, 'weight1'] = 0
        df.loc[df['year'] < 2013, 'weight2'] = 0

        # Make sure dates too far in the future are zero weighted
        df.loc[df['year'] > 2015, 'weight0'] = 0
        df.loc[df['year'] > 2015, 'weight1'] = 0
        df.loc[df['year'] > 2015, 'weight2'] = 0

    def build_reco_old(self):
        """
        Build best hotels given keys
        For performance reasons, apply method uses raw=True to pass ndarrays and speed things up
        However it seems that pandas does not use ndarrays unless all data types are the same in the DataFrame
        Mixed type are still passed in a pd.Series object !!
        SO if you pass int + float + string pandas will pass a pd.Series !!!! and performance will drop dramatically
        Not sure if this is a bug or if it is according to plan...
        So as a last resort I finally used numpy apply_along_axis function to keep performance
        Performance for hotel_country key is at 3.3 minutes
        :return:
        """
        if self.the_keys is None:
            raise ValueError('No key has been specified')
        # Init hotel reco
        nb_lines = 0
        nb_bookings = 0
        start = time.time()
        # Choose which file to read to build the recommendations
        if self.do_submission:
            input_file = 'input/train.csv'
        else:
            input_file = 'input/trn_data.csv'

        # Read the files by chunks
        for k, df in enumerate(pd.read_csv(input_file, chunksize=100000, iterator=True)):
            # Compute weights
            self.update_weights(df=df, ref_type='checkin')
            # weight 0 gives less importance to older records regardless they are clicks or bookings
            df['weight'] = df['weight0']
            # 1st weight strategy gives more weight to bookings than clicks
            if self.weight_strategy == 1:
                df['weight'] = df['weight1']
            # 2nd weight strategy does not reduce old events importance but weight bookings more than clicks
            if self.weight_strategy == 2:
                df['weight'] = df['weight2']

            # This call takes 10 times slower than the selected call
            # raw=False uses pd.Series
            # df[the_keys + ['append', 'hotel_cluster']].dropna().apply(lambda row: self.update_reco(row),
            #                                                           axis=1, raw=False)

            # This call is a lot quicker if data types are not mixed i.e. all floats or all ints
            # In this case raw=True uses ndarrays
            # However for some reason if we have mixed data types int and float raw=True reverts to
            # using pd.Series !?!
            # df[self.the_keys + ['weight', 'hotel_cluster']].dropna().apply(lambda row: self.update_reco_raw(row),
            #                                                                axis=1, raw=True)

            # As a last resort I used numpy apply_along_axis with a prior conversion of pd.DataFrame to ndarray using
            # .values attribute that gives a view of the DataFrame
            the_rows = self.the_keys + ['weight', 'hotel_cluster']
            np.apply_along_axis(func1d=self.update_reco_raw, axis=1, arr=df[the_rows].dropna().values)

            nb_lines += len(df)
            nb_bookings += df['is_booking'].sum()
            if nb_lines % 1000000 == 0:
                print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))

                # if k > 2:
                #      break
        print(nb_lines, nb_bookings)

    @staticmethod
    def get_df_as_np(df, rows):
        return df[rows].dropna().values

    def build_reco(self):
        """
        Build best hotels given keys
        For performance reasons, apply method uses raw=True to pass ndarrays and speed things up
        However it seems that pandas does not use ndarrays unless all data types are the same in the DataFrame
        Mixed type are still passed in a pd.Series object !!
        SO if you pass int + float + string pandas will pass a pd.Series !!!! and performance will drop dramatically
        Not sure if this is a bug or if it is according to plan...
        So as a last resort I finally used numpy apply_along_axis function to keep performance
        Performance for hotel_country key is at 3.3 minutes
        :return:
        """
        if self.the_keys is None:
            self.the_keys = [
                ('leak', ['user_location_city', 'orig_destination_distance'], 0),
                ('uid_dest', ['user_id', 'srch_destination_id'], 1),
                ('dest_mkt_pack', ['srch_destination_id', 'hotel_market', 'is_package'], 1),
                ('dest', ['srch_destination_id'], 1),
                ('hot_co_mkt', ['hotel_country', 'hotel_market'], 2),
                ('hot_co', ['hotel_country'], 2),
            ]

        nb_lines = 0
        nb_bookings = 0
        start = time.time()
        # Choose which file to read to build the recommendations
        if self.do_submission:
            input_file = 'input/train.csv'
        else:
            input_file = 'input/trn_data.csv'

        # Read the files by chunks
        for k, df in enumerate(pd.read_csv(input_file, chunksize=250000, iterator=True)):
            # Compute weights
            self.update_weights(df=df, ref_type='checkin')

            # Loop over keys
            for k_, (name_, keys_, w_type_) in enumerate(self.the_keys):
                # Choose the weight according to w_type_
                # weight 0 gives less importance to older records regardless they are clicks or bookings
                df['weight'] = df['weight0']
                # 1st weight strategy gives more weight to bookings than clicks
                if w_type_ == 1:
                    df['weight'] = df['weight1']
                # 2nd weight strategy does not reduce old events importance but weight bookings more than clicks
                if w_type_ == 2:
                    df['weight'] = df['weight2']

                # This call takes 10 times slower than the selected call
                # raw=False uses pd.Series
                # df[the_keys + ['append', 'hotel_cluster']].dropna().apply(lambda row: self.update_reco(row),
                #                                                           axis=1, raw=False)

                # This call is a lot quicker if data types are not mixed i.e. all floats or all ints
                # In this case raw=True uses ndarrays
                # However for some reason if we have mixed data types int and float raw=True reverts to
                # using pd.Series !?!
                # df[self.the_keys + ['weight', 'hotel_cluster']].dropna().apply(lambda row: self.update_reco_raw(row),
                #                                                                axis=1, raw=True)

                # As a last resort use numpy apply_along_axis with a prior conversion of pd.DataFrame to ndarray using
                # .values attribute thatgives a view of the DataFrame
                the_rows = keys_ + ['weight', 'hotel_cluster']
                # Improve process speed using groupby function
                gp_df = df[the_rows].dropna().groupby(keys_ + ['hotel_cluster']).sum().reset_index()
                # Make sure rows are in the correct order
                gp_df = gp_df[the_rows].values
                # gp_df = df[the_rows].dropna().values
                for i in range(len(gp_df)):
                    self.update_reco_raw(gp_df[i, :], len_keys=len(keys_), key_nb=k_)

                # print(gp_df.sort_values(by='hotel_cluster'))
                # print(len(df), len(gp_df))
                # np.apply_along_axis(func1d=self.update_reco_raw,
                #                     axis=1,
                #                     arr=df[the_rows].dropna().values,
                #                     # arr=self.get_df_as_np(gp_df, the_rows),
                #                     # arr=gp_df.values,
                #                     len_keys=len(keys_),
                #                     key_nb=k_)

            nb_lines += len(df)
            nb_bookings += df['is_booking'].sum()
            if nb_lines % 1000000 == 0:
                print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))

            # if k > 1:
            #     print("Interrupted build for quick testing")
            #     break

        print(nb_lines, nb_bookings)

    def apply_reco_to_dataset(self, row, indices=None):
        """

        :param row: key values
        :param indices: list of indices to access keys directly in row
        :return: recommendations
        """
        self.validation_recos += 1
        len_recos = 0
        recos = []
        # Loop over registered keys
        k_ = 0
        while (k_ < len(self.the_keys)) and (len_recos < self.nb_recos):
        # for k_ in range(len(self.the_keys)):
            # Compute the key using the indices
            dict_key = tuple(row[indices[k_]])
            # Check if dict_key is in the k_ th defaultdict
            if dict_key in self.best_hotels[k_]:
                # Get the list of hotels for the keys
                d = self.best_hotels[k_][dict_key]
                # Sort the hotels to find the best hotels
                top_items = nlargest(self.nb_recos, sorted(d.items()), key=itemgetter(1))
                # Loop over best hotels to fill remaining places
                for i in range(len(top_items)):
                    # Check if we have the number of required reco
                    if len(recos) == self.nb_recos:
                        break
                    # Check if reco is already in
                    if top_items[i][0] in recos:
                        continue
                    # Add current recommendation for this line
                    recos.append(top_items[i][0])
                    # update number of times the key has been used
                    self.validation_stats[k_][len_recos][0] += 1
                    # update the number of times this reco is the target hotel cluster at this place
                    if not self.do_submission:
                        # Last element in row is the hotel_cluster
                        if top_items[i][0] == row[-1]:
                            self.validation_stats[k_][len_recos][1] += 1
                    len_recos += 1
            # go to the next key
            k_ += 1

        # Return the list of recommendations in the required format (i.e. space separated)
        str_recos = [str(int(reco)) for reco in recos]
        return ' '.join(str_recos)

    def assign_reco_to_validation(self):
        # Display a few stuff
        # for kk in self.best_hotels.keys():
        #     print(len(self.best_hotels[kk].keys()))
        #     for j, kkk in enumerate(self.best_hotels[kk].keys()):
        #         print("    ->", self.best_hotels[kk][kkk])
        #         if j > 10:
        #             break

        start = time.time()
        nb_lines = 0
        # Init validation stats :
        # Key is the place where reco has been set : 0 to NB_RECOS - 1
        # 1st item is the number of recos filled at this place
        # 2nd item is the number of time this reco has been exact
        # read data using pandas by chunk
        first_save = True

        # Gather all requested keys
        all_keys = set()
        for _, keys_, _ in self.the_keys:
            all_keys = all_keys.union(keys_)
        all_keys = list(all_keys)

        # Now find key indices in all_keys for each key
        indices = []
        for _, keys_, _ in self.the_keys:
            indices.append([i for j in range(len(keys_)) for i in range(len(all_keys)) if all_keys[i] == keys_[j]])

        for df in pd.read_csv('input/val_data.csv', chunksize=100000, iterator=True):
            the_rows = all_keys + ['hotel_cluster']

            np_df = df[the_rows].values
            df['recos'] = ""
            np_recos = df.recos.values
            for i in range(len(np_df)):
                np_recos[i] = self.apply_reco_to_dataset(np_df[i, :], indices)
            df['recos'] = np_recos

            # df['recos'] = np.apply_along_axis(func1d=self.apply_reco_to_dataset,
            #                                   axis=1,
            #                                   arr=df[the_rows].values,
            #                                   indices=indices)

            df['id'] = df.index.values

            nb_lines += len(df)
            if nb_lines % 100000 == 0:
                print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))

            if first_save:
                df[['id', 'recos']].to_csv('submission/oof_' + self.name + '.csv', index=False)
                first_save = False
            else:
                df[['id', 'recos']].to_csv('submission/oof_' + self.name + '.csv', mode='a', header=None, index=False)

    def assign_reco_to_submission(self):
        # Init some data for display purposes
        start = time.time()
        nb_lines = 0

        # read data using pandas by chunk
        first_save = True

        # Gather all requested keys
        all_keys = set()
        for _, keys_, _ in self.the_keys:
            all_keys = all_keys.union(keys_)
        all_keys = list(all_keys)

        # Now find key indices in all_keys for each key
        indices = []
        for _, keys_, _ in self.the_keys:
            indices.append([i for j in range(len(keys_)) for i in range(len(all_keys)) if all_keys[i] == keys_[j]])

        for k, df in enumerate(pd.read_csv('input/test.csv', chunksize=100000, iterator=True)):

            df['hotel_cluster'] = np.apply_along_axis(func1d=self.apply_reco_to_dataset,
                                                      axis=1,
                                                      arr=df[all_keys].values,
                                                      indices=indices)

            nb_lines += len(df)
            if nb_lines % 500000 == 0:
                print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))

            if first_save:
                df[['id', 'hotel_cluster']].to_csv('submission/sub_' + self.name + '.csv',
                                                   index=False)
                first_save = False
            else:
                df[['id', 'hotel_cluster']].to_csv('submission/sub_' + self.name + '.csv',
                                                   mode='a',
                                                   header=None,
                                                   index=False)
                pd.DataFrame().to_csv()

    def assign_reco_to_dataset(self):
        if self.do_submission:
            self.assign_reco_to_submission()
        else:
            self.assign_reco_to_validation()

    def display_stats(self):
        # if self.do_submission:
        #     print("No statistics available on submission data")
        #     return None

        # make and display header
        header = ""
        for i in range(self.nb_recos):
            header += " HITS@{0:d}   %USE   %ACC   ".format(i + 1)
        print("{0:<20s} {1:s}".format("", header))

        # Print individual stats
        map = 0
        total_hits = np.zeros(self.nb_recos)
        total_use = np.zeros(self.nb_recos)
        for k_, (name_, keys_, w_type_) in enumerate(self.the_keys):
            str_res = ""

            for i in range(self.nb_recos):
                str_res += "{1:.5f} {2:6.2f} {3:6.2f} | ".format(
                    i,
                    self.validation_stats[k_][i][1] / self.validation_recos,
                    100 * self.validation_stats[k_][i][0] / self.validation_recos,
                    100 * self.validation_stats[k_][i][1] / (1e-5 + self.validation_stats[k_][i][0]),
                )
                map += (self.validation_stats[k_][i][1] / self.validation_recos) / (i + 1)
                total_hits[i] += self.validation_stats[k_][i][1]
                total_use[i] += self.validation_stats[k_][i][0]

            print("{0:<20s} {1:s}".format(name_.upper(), str_res))

        # Print full recap
        str_res = ""
        for i in range(self.nb_recos):
            str_res += "{1:.5f} {2:6.2f} {3:6.2f} | ".format(
                i,
                total_hits[i] / self.validation_recos,
                100 * total_use[i] / self.validation_recos,
                100 * total_hits[i] / (1e-5 + total_use[i]),
            )
        print("{0:<20s} {1:s}".format("TOTAL", str_res))

        print("Total MAP = ", map)
