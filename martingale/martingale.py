import numpy as np


def get_spin_result(win_prob):
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def realistic_strategy_simulator(win_prob, bank_roll=np.inf, it=1001):
    winnings = np.zeros(it)
    episode_winnings = 0
    i = 1

    while episode_winnings < 80 and i < it and bank_roll > 0:
        won = False
        bet_amount = 1

        while not won and i < it and bank_roll > 0:
            won = get_spin_result(win_prob)

            if won:
                episode_winnings += bet_amount
                bank_roll += bet_amount

            else:
                episode_winnings -= bet_amount
                bank_roll -= bet_amount
                bet_amount *= 2

                if bet_amount > bank_roll:
                    bet_amount = bank_roll

            winnings[i] = episode_winnings
            i += 1

    winnings[i:] = winnings[i - 1]

    return winnings
