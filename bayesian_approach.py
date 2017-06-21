"""
Build recommendations based on past data
Data files contain :
'date_time' -> Used to split data between train and test for local validation
'site_name' -> Unused
'posa_continent' -> Unused
'user_location_country' -> Unused
'user_location_region' -> Unused
'user_location_city' -> Used as part of a leakage
'orig_destination_distance' -> Used as part of a leakage
'user_id' -> Used for bayes group 2
'is_mobile' -> Unused
'is_package' -> Used in group 3
'channel' -> Unused
'srch_ci' -> Used for weighting purposes
'srch_co' -> Used for weighting purposes
'srch_adults_cnt' -> Unused
'srch_children_cnt' -> Unused
'srch_rm_cnt' -> Unused
'srch_destination_id' -> used in groups 2 and 3
'srch_destination_type_id' -> Unused
'is_booking' -> Used for weighting purposes
'cnt' -> Unused
'hotel_continent' -> Unused
'hotel_country' -> Used as part of groups 4 and 5
'hotel_market' -> Used in group 3
'hotel_cluster' -> Target

Bayesian recommendations are based on 5 groupings :
A leak : (user_location_city, orig_destination_distance) -> Local = 0.25968 / LB =
Then   : (srch_destination_id, hotel_market, is_package) -> Local = 0.31213 / LB =
Then   : (hotel_country, hotel_market)                   -> Local = 0.24780 / LB =
Then   : (user_id, srch_destination_id)                  -> Local = 0.05631 / LB =
Finally (hotel_country)                                  -> Local = 0.14118 / LB =

With the following stats, the most important number is ACC (accuracy)
Bayesian analysis: submission:False, keys:['user_location_city', 'orig_destination_distance']
  MAP@1  %ACC     MAP@2  %ACC     MAP@3  %ACC     MAP@4  %ACC     MAP@5  %ACC
0.25010 85.54 | 0.01841 31.43 | 0.00107 15.85 | 0.00005 10.15 | 0.00000  5.26
Bayesian analysis: submission:False, keys:['user_id', 'srch_destination_id']
  MAP@1  %ACC     MAP@2  %ACC     MAP@3  %ACC     MAP@4  %ACC     MAP@5  %ACC
0.04592 28.85 | 0.01403 13.73 | 0.00634  8.95 | 0.00344  6.77 | 0.00199  5.38
Bayesian analysis: submission:False, keys:['srch_destination_id', 'hotel_market', 'is_package']
  MAP@1  %ACC     MAP@2  %ACC     MAP@3  %ACC     MAP@4  %ACC     MAP@5  %ACC
0.19160 19.29 | 0.12053 12.23 | 0.09044  9.26 | 0.07330  7.58 | 0.05898  6.17
Bayesian analysis: submission:False, keys:['hotel_country', 'hotel_market']
  MAP@1  %ACC     MAP@2  %ACC     MAP@3  %ACC     MAP@4  %ACC     MAP@5  %ACC
0.14480 14.48 | 0.09625  9.63 | 0.08078  8.08 | 0.06692  6.69 | 0.05610  5.61
Bayesian analysis: submission:False, keys:['hotel_country']
  MAP@1  %ACC     MAP@2  %ACC     MAP@3  %ACC     MAP@4  %ACC     MAP@5  %ACC
0.08245  8.24 | 0.05837  5.84 | 0.04207  4.21 | 0.03628  3.63 | 0.03226  3.23
"""
import sys
import getopt
import time
import pandas as pd
from utils.bayesian_expedia_recommender import BayesianExpediaReco
import cProfile

NB_RECOS = 5

unused_cols = ['site_name',
               'posa_continent',
               'user_location_country',
               'user_location_region',
               'is_mobile',
               'channel',
               'srch_adults_cnt',
               'srch_children_cnt',
               'srch_rm_cnt',
               'srch_destination_type_id',
               'cnt', 'hotel_continent',
               ]


def usage():
    print("usage :")
    print('-m : processing mode, can be val for validation or sub for submission')
    print('-k : comma separated list of keys')
    print('python bayesian_approach.py -mval --keys=user_location_city,orig_destination_distance')
    print('python bayesian_approach.py -msub --keys=user_location_city,orig_destination_distance')


def check_args(argv):
    """
    Check program arguments
    :param argv: command line arguments
    :return: do_submission (Boolean) and the_keys lost of fields used to build hotel recommendation
    """
    the_keys_ = None
    do_submission_ = None
    build_training_files_ = False
    name_ = 'standard'
    weight_type_ = None

    try:
        opts, args = getopt.getopt(argv, "m:k:n:w:b", ["mode=", "keys=", "name=", "weight="])
        for opt, arg in opts:
            if opt == '-h':
                usage()
                sys.exit()
            elif opt in ("-m", "--mode"):
                if arg in ['val', 'validation']:
                    do_submission_ = False
                elif arg in ['submission', 'sub']:
                    do_submission_ = True
                else:
                    usage()
                    sys.exit(2)
            elif opt in ("-k", "--keys"):
                the_keys_ = arg.split(',')
                if len(the_keys_) == 0:
                    usage()
                    sys.exit(2)
            elif opt in ("-n", "--name"):
                name_ = arg
            elif opt in ("-w", "--weight"):
                try:
                    weight_type_ = int(arg)
                except TypeError:
                    raise TypeError('Weight type must be an int')
            elif opt in ("-b", "--build"):
                build_training_files_ = True
            else:
                usage()
                sys.exit()
        if len(opts) == 0:
            usage()
            sys.exit(2)
        #if (the_keys_ is None) and (not build_training_files_):
        #    usage()
        #    sys.exit(2)

    except getopt.GetoptError:
        usage()
        sys.exit(2)

    return do_submission_, the_keys_, weight_type_, name_, build_training_files_


def build_files_for_training():
    start = time.time()
    nb_lines = 0
    # read data using pandas by chunk
    trn_first = True
    val_first = True
    for k, df in enumerate(pd.read_csv('input/train.csv', chunksize=200000, iterator=True)):
        df['pd_date_time'] = pd.to_datetime(df['date_time'])
        df['year'] = df['pd_date_time'].dt.year.astype(int)
        df['month'] = df['pd_date_time'].dt.month.astype(int)
        df['dom'] = df['pd_date_time'].dt.day.astype(int)
        del df['pd_date_time']
        df.drop(unused_cols, axis=1, inplace=True)

        trn_filter = (df.year == 2013) | ((df.year == 2014) & (df.month <= 4))
        val_filter = (df.year == 2014) & (df.month > 4) & (df.is_booking == 1)

        if trn_first:
            print(df.columns)
            df.loc[trn_filter].to_csv('input/trn_data.csv')
            trn_first = False
        else:
            df.loc[trn_filter].to_csv('input/trn_data.csv', mode='a', header=None)

        if val_first:
            df.loc[val_filter].to_csv('input/val_data.csv')
            val_first = False
        else:
            df.loc[val_filter].to_csv('input/val_data.csv', mode='a', header=None)

        nb_lines += len(df)
        print("nb_lines %10d in %5.1f " % (nb_lines, (time.time() - start) / 60))


if __name__ == '__main__':

    # Check arguments
    do_submission, the_keys, weight_type, name, build_training_files = check_args(sys.argv[1:])
    if build_training_files:
        build_files_for_training()
    else:
        print('Bayesian analysis: submission:%s, keys:%s'
              % (do_submission, the_keys))
        ber = BayesianExpediaReco(nb_recos=NB_RECOS,
                                  do_submission=do_submission,
                                  weight_type=weight_type,
                                  name=name)
        if the_keys:
            ber.register_key(the_name=name, the_key=the_keys, the_weight=weight_type)
        # Get recommendations from train data
        # cProfile.run('ber.build_reco()')
        print("Building reco")
        ber.build_reco()
        # Assign recommendations
        print("Assigning RECO")
        ber.assign_reco_to_dataset()
        # cProfile.run('ber.build_reco()')
        # Display validation stats
        ber.display_stats()
