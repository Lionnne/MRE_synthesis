#!/usr/bin/env python3

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

from tofgatir.trainers import TrainerBuilder

builder = TrainerBuilder(args)
trainer = builder.build()
trainer.start()
trainer.run()
trainer.close()