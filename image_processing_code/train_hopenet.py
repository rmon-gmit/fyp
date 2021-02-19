import datasets
import hopenet

PROJECT_DIR = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\"

AFLW2000_DATA_DIR = "C:\\Users\\rossm\\AFLW2000\\"
AFLW2000_TEST_SAVE_DIR = PROJECT_DIR + "data\\aflw2000_test\\"

BIWI_DATA_DIR = "C:\\Users\\rossm\\kinect_head_pose_db\\hpdb\\"
BIWI_TEST_SAVE_DIR = PROJECT_DIR + "data\\biwi_test\\"

# MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\models\\alexnet_biwi_25_epoch.h5"
MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\models\\vgg16_aflw_25_epoch_flipped.h5"

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16  # 16
EPOCHS = 25  # 25

# dataset = datasets.Biwi(BIWI_DATA_DIR, '\\filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.8)
dataset = datasets.AFLW2000(AFLW2000_DATA_DIR, '\\filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

net = hopenet.HopeNet(dataset, INPUT_SIZE, BIN_NUM, BATCH_SIZE, MODEL_PATH)

net.train(MODEL_PATH, epochs=EPOCHS, load_weight=False)
