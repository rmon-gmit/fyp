import datasets
import hopenet

PROJECT_DIR = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\image_processing_code\\"

AFLW2000_DATA_DIR = PROJECT_DIR + "data\\AFLW2000\\"
AFLW2000_TEST_SAVE_DIR = PROJECT_DIR + "data\\aflw2000_test\\"

BIWI_DATA_DIR = "C:\\Users\\rossm\\kinect_head_pose_db\\hpdb\\"
BIWI_TEST_SAVE_DIR = PROJECT_DIR + "data\\biwi_test\\"

MODEL_PATH = "C:\\Users\\rossm\\OneDrive - GMIT\\Year 4\\Final Year Project\\models\\vgg16_biwi_7_epoch.h5"

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 7  # 20

dataset = datasets.Biwi(BIWI_DATA_DIR, '\\filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.8)

net = hopenet.HopeNet(dataset, INPUT_SIZE, BIN_NUM, BATCH_SIZE, MODEL_PATH)

net.train(MODEL_PATH, max_epochs=EPOCHS, load_weight=False)
