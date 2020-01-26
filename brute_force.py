nb_steps = 50
increment = 1.0 / nb_steps

max_value = -1000
best_defence_global = []
for i_a1 in range(nb_steps):
    a1 = i_a1 * increment
    for i_a2 in range(nb_steps):
        a2 = i_a2 * increment
        a3 = 1 - a2 - a1
        if a3 < 0 or a3 > 1:
            continue
        print("Attacker uses {}, {}, {}".format(a1, a2, a3))
        min_value = 1000
        best_defence = []
        for i_d1 in range(nb_steps):
            d1 = i_d1 * increment
            for i_d2 in range(nb_steps):
                d2 = i_d2 * increment
                d3 = 1 - d2 - d1
                if d3 < 0 or d3 > 1:
                    continue
                # print("Defender uses {}, {}, {}".format(d1, d2, d3))
                value = a1 * (d1 - d3) + a2 * (d2 + d3 - d1) + a3 * (d3 - d2)
                if value < min_value:
                    min_value = value
                    best_defence = [d1, d2, d3]
        print(
            "Best defence is {}, {}, {}. Expected value is {}".format(
                best_defence[0], best_defence[1], best_defence[2], min_value
            )
        )
        if min_value > max_value:
            max_value = min_value
            best_attack = [a1, a2, a3]
            best_defence_global = best_defence
print(
    "Done !\nBest attack: {}\nBest defence against it: {}\nExpected value: {}".format(
        best_attack, best_defence_global, max_value
    )
)

