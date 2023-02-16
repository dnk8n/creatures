import time
from random import random, shuffle
from threading import Thread

from utils.logging import log

num_creatures = 10
num_rounds = 3
creatures = [str(c).zfill(2) for c in range(num_creatures)]


def fight(creature1, creature2, round_idx):
    log.info(f"BEGIN: {creature1} vs {creature2} - Round {round_idx+1}")
    time.sleep(random() * 5)
    log.info(f"END: {creature1} vs {creature2} - Round {round_idx+1}")


for round_idx in range(num_rounds):
    log.info(f"Round {round_idx+1} started!")
    shuffle(creatures)
    threads = [
        Thread(target=fight, args=(creatures[i], creatures[i + 1], round_idx))
        for i in range(0, num_creatures, 2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log.info(f"Round {round_idx+1} complete!\n\n")
    time.sleep(3)
