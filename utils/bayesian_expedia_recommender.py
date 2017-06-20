"""
Using no weighting strategy i.e. weight = 1 : LB score is : Private 0.49165 / Public 0.49482
Using standard weighting strategy           : LB score is : Private 0.49270 / Public 0.49603
Using weighting strategy 1                  : LB score is : Private 0.49550 / Public 0.49906
Using weighting strategy 2                  : LB score is : Private 0.49184 / Public 0.49522
Using different strategies for each key     : LB score is : Private 0.49 / Public 0.49
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
                 the_keys=None,
                 do_submission=False,
                 weight_type=None,
                 name='standard'):
        self.nb_recos = nb_recos
        self.best_hotels = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.validation_stats = defaultdict(lambda: {x: [0, 0] for x in range(self.nb_recos)})
        self.the_keys = the_keys
        self.len_keys = 0
        if self.the_keys is not None:
            self.len_keys = len(self.the_keys)
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

    def get_key(self, row):
        # print(row)
        return tuple(row[:self.len_keys])

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
        for k, df in enumerate(pd.read_csv(input_file, chunksize=100000, iterator=True)):
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
                # print(the_rows)
                np.apply_along_axis(func1d=self.update_reco_raw,
                                    axis=1,
                                    arr=df[the_rows].dropna().values,
                                    len_keys=len(keys_),
                                    key_nb=k_)

            nb_lines += len(df)
            nb_bookings += df['is_booking'].sum()
            if nb_lines % 1000000 == 0:
                print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))

            # if k > 2:
            #     print("Interrupted build for quick testing")
            #     break

        print(nb_lines, nb_bookings)

    def apply_reco_to_dataset(self, row, indices=None):
        """
        Validation applies to row['year'] == 2013) or ((row['year'] == 2014) and (row['month'] <= 4)
        :param row: key values
        :return: recommendations
        """
        self.validation_recos += 1
        # print('val : ', self.validation_recos)
        # print('indices : ', indices)
        len_recos = 0
        recos = []
        # print('row : ', row)
        # Loop over registered keys
        # print(self.the_keys)
        for k_, (name_, keys_, w_type_) in enumerate(self.the_keys):
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
                    # self.validation_stats[len_recos][0] += 1
                    self.validation_stats[k_][len_recos][0] += 1
                    # update the number of times this reco is the target hotel cluster at this place
                    # if top_items[i][0] == row['hotel_cluster']:
                    if not self.do_submission:
                        # Last element in row is the hotel_cluster
                        if top_items[i][0] == row[-1]:
                            # self.validation_stats[len_recos][1] += 1
                            self.validation_stats[k_][len_recos][1] += 1
                    len_recos += 1
        # Return the list of recommendations in the required format (i.e. space separated)
        str_recos = [str(int(reco)) for reco in recos]
        return ' '.join(str_recos)

    def assign_reco_to_validation(self):
        # TODO accelerate process
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
            df['recos'] = np.apply_along_axis(func1d=self.apply_reco_to_dataset,
                                              axis=1,
                                              arr=df[the_rows].values,
                                              indices=indices)

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
            # df['pd_date_time'] = pd.to_datetime(df['date_time'])
            # df['year'] = df['pd_date_time'].dt.year
            # df['month'] = df['pd_date_time'].dt.month
            # df['dom'] = df['pd_date_time'].dt.day

            # df['hotel_cluster'] = df[self.the_keys].apply(lambda row: self.apply_reco_to_dataset(row), axis=1, raw=True)
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
        if self.do_submission:
            print("No statistics available on submission data")
            return None

        print("Results for keys %s" % self.the_keys)

        score = 0
        for i in range(self.nb_recos):
            score += self.validation_stats[i][1] / ((i + 1) * self.validation_recos)
            print("%% Hits @ place %d : %.5f"
                  % (i, self.validation_stats[i][1] / self.validation_recos))
        print(score)

        header = ""
        str_res = ""
        for i in range(self.nb_recos):
            header += "  MAP@{0:d}  %ACC   ".format(i + 1)
            str_res += "{1:.5f} {2:5.2f} | ".format(
                i,
                self.validation_stats[i][1] / self.validation_recos,
                100 * self.validation_stats[i][1] / self.validation_stats[i][0]
            )
        print(header)
        print(str_res)

    def display_stats2(self):
        if self.do_submission:
            print("No statistics available on submission data")
            return None

        # make and display header
        header = ""
        for i in range(self.nb_recos):
            header += " HITS@{0:d}  %USE  %ACC   ".format(i + 1)
        print("{0:<20s} {1:s}".format("", header))

        # Print individual stats
        map = 0
        for k_, (name_, keys_, w_type_) in enumerate(self.the_keys):
            str_res = ""

            for i in range(self.nb_recos):
                str_res += "{1:.5f} {2:5.2f} {3:5.2f} | ".format(
                    i,
                    self.validation_stats[k_][i][1] / self.validation_recos,
                    100 * self.validation_stats[k_][i][0] / self.validation_recos,
                    100 * self.validation_stats[k_][i][1] / (1e-5 + self.validation_stats[k_][i][0]),
                )
                map += (self.validation_stats[k_][i][1] / self.validation_recos) / (i + 1)

            print("{0:<20s} {1:s}".format(name_.upper(), str_res))
        print("Total MAP = ", map)
