import os

# DIR
DIR_HOME = os.path.abspath(os.path.dirname(os.path.abspath("")))
DIR_MODEL = os.path.join(DIR_HOME, 'models')
DIR_DATA = os.path.join(DIR_HOME, 'data')
DIR_LOG = os.path.join(DIR_HOME, 'logs')
DIR_SUBMISSIONS = os.path.join(DIR_DATA, 'submission')
DIR_INTERIM = os.path.join(DIR_DATA, 'interim')
DIR_FEATURE = os.path.join(DIR_DATA, 'features')
DIR_FIGURE = os.path.join(DIR_DATA, 'figures')
DIR_RAW = os.path.join(DIR_DATA, 'raw')
DIR_INPUT = os.path.join(DIR_RAW, 'input')
DIR_IMAGE = os.path.join(DIR_INPUT, 'images')
DIR_CROPS = os.path.join(DIR_INPUT, 'crops')
DIR_META = os.path.join(DIR_INPUT, 'atmaCup22_2nd_meta')

# FILE
FILE_TRAIN_META = os.path.join(DIR_META, 'train_meta.csv')
FILE_TEST_META = os.path.join(DIR_META, 'test_meta.csv')
FILE_SAMPLE_SUBMISSION = os.path.join(DIR_INPUT, 'sample_submission.csv')


# CONFIG


