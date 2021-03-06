
import torch 

EPOCHS          = 20
LEARNING_RATE   = 0.002
L1_LAMBDA       = 100
BATCH_SIZE      = 32
DEVICE          = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS     = 8
IMG_CHANNELS    = 3
FEATURES        = 3
NUM_WORKER      = True


