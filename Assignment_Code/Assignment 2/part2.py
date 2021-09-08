import numpy as np
from matplotlib import pyplot as plt
from som_net import SOM
import pandas as pd
import math
from matplotlib.path import Path
import matplotlib.patches as patches


# 4.1
def animal_ordering():
    data = np.loadtxt('Data/animals.dat', delimiter=",", dtype=int)
    data = np.reshape(data, (32, 84))

    names = np.loadtxt('Data/animalnames.txt', dtype=str)
    for i, name in enumerate(names):
        names[i] = name[1:-1]

    weight_matrix = np.random.rand(100, 84)

    # 20 epochs, 0.2 eta, 50 initial neighbourhood size
    som = SOM(data, weight_matrix, 50, 0.2, 50)
    predictions = som.train()

    for i, p in enumerate(predictions):
        p[1] = names[i]

    predictions.sort(key=lambda x: x[0])

    x_list = []
    for pred in predictions:
        x_list.append(pred[0])

    fig, ax = plt.subplots()
    ax.scatter(x_list, len(predictions) * [0])

    for i, p in enumerate(predictions):
        if i % 2 == 0:
            ax.annotate(p[1], (x_list[i], 0), rotation=45,
                        va="bottom", ha="center")
        else:
            ax.annotate(p[1], (x_list[i], 0), rotation=45,
                        va="top", ha="center")

    #plt.scatter(x_list, len(predictions) * [0])
    plt.show()


# 4.2
def cyclic_tour():

    def plot_tour(inputs, weights):
        outputs = []
        for cityNdx in range(10):
            similarity = np.sum(
                np.square(inputs[cityNdx, :] - weights), axis=1)
            indice = np.argmin(similarity)
            outputs.append(weights[indice, :])
        outputs.append(inputs[0, :])
        outputs = np.asarray(outputs)
        outputs = np.vstack([outputs[:, :], outputs[0, :]])
        plt.scatter(inputs[:, 0], inputs[:, 1], color="red")
        plt.plot(outputs[:, 0], outputs[:, 1], color='green')
        plt.show()

    data = np.loadtxt('Data/cities.dat', delimiter=",", dtype=float)
    data = np.reshape(data, (10, 2))

    for i in range(10):
        weight_matrix = np.random.uniform(0, 1, (10, 2))
        som = SOM(data, weight_matrix, 100, 0.1, 1)
        predictions, weights = som.train("cyclic")
        plot_tour(data, weights)


# 4.3
def mp_votes():
    all_mp = []

    vote_data = np.loadtxt('Data/votes.dat', delimiter=",", dtype=float)
    vote_data = np.reshape(vote_data, (349, 31))

    # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    # Use some color scheme for these different groups
    party_data = np.loadtxt('Data/mpparty.dat', dtype=int)

    # Coding: Male 0, Female 1
    gender_data = np.loadtxt('Data/mpsex.dat', dtype=int)

    disctrict_data = np.loadtxt('Data/mpdistrict.dat', dtype=int)

    name_data = np.genfromtxt(
        'Data/mpnames.txt', dtype=str, encoding='latin-1', delimiter="\n")

    for i in range(349):
        # Structure: name, party, gender, district
        all_mp.append([name_data[i], party_data[i],
                       disctrict_data[i], gender_data[i]])

    weight_matrix = np.random.rand(100, 31)

    som = SOM(vote_data, weight_matrix, 10, 0.1, 50)
    predictions = som.train('votes')
    predictions = np.array(predictions)

    npgrid = mp_to_grid(predictions[:, 0])
    unique, counts = np.unique(npgrid, return_counts=True, axis=0)
    counts = counts.reshape(unique.shape[0], 1)
    compact_grid = np.column_stack((unique, counts))

    names = (pd.DataFrame(name_data)
             .rename(columns={0: 'names'})
             .reindex(predictions[:, 1])
             .reset_index()
             .rename(columns={'index': 'id'}))
    gender = (pd.DataFrame(gender_data)
                .rename(columns={0: 'gender'})
                .reindex(predictions[:, 1])
                .reset_index()
                .rename(columns={'index': 'id'}))
    district = (pd.DataFrame(disctrict_data)
                .rename(columns={0: 'district'})
                .reindex(predictions[:, 1])
                .reset_index()
                .rename(columns={'index': 'id'}))
    party = (pd.DataFrame(party_data)
             .rename(columns={0: 'party'})
             .reindex(predictions[:, 1])
             .reset_index()
             .rename(columns={'index': 'id'}))

    grid = (pd.DataFrame(npgrid)
            .rename(columns={0: 'x', 1: 'y'}))

    grid['original_output'] = predictions[:, 0]
    grid['id'] = predictions[:, 1]

    grid = (grid.merge(names, how='left', on='id')
                .merge(gender, how='left', on='id')
                .merge(district, how='left', on='id')
                .merge(party, how='left', on='id')
                .merge(pd.DataFrame(compact_grid).rename(columns={0: 'x', 1: 'y', 2: 'count'}), on=['x', 'y'], how='left'))

    # gender visuals
    for value in grid['gender'].unique():
        df = grid[grid.gender == value]
        plt.scatter(df['x'], df['y'], s=df['count']*5, c='red', alpha=0.7)
        axes = plt.gca()
        axes.set_xlim([-1, 11])
        axes.set_ylim([-1, 11])
        print(len(df))
        plt.show()

    # party visuals
    # Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
    party_colors = {
        0: "grey",
        1: "#3c0cc2",
        2: "#15c6e6",
        3: "#e61515",
        4: "#f00eb7",
        5: "#269612",
        6: "#967712",
        7: "#2bff24"
    }

    party_names = {
        0: "no party",
        1: "M",
        2: "FP",
        3: "S",
        4: "V",
        5: "MP",
        6: "KD",
        7: "C"
    }

    for value in sorted(grid['party'].unique()):
        df = grid[grid.party == value]
        plt.scatter(df['x'], df['y'], s=df['count']*2,
                    c=party_colors[value], alpha=1, label=party_names[value])

        #plt.title('party number {}'.format(value))

    axes = plt.gca()
    axes.set_xlim([-1, 11])
    axes.set_ylim([-1, 11])
    axes.legend(loc='best', frameon=False)
    plt.show()

    # district visuals

    district_colors = {
        1: "#FF0049",
        2: "#FF008B",
        3: "#FF00F7",
        4: "#E400FF",
        5: "#AA00FF",
        6: "#6C00FF",
        7: "#2E00FF",
        8: "#3E39A0",
        9: "#7D79D8",
        10: "#79D3D8",
        11: "#33E7F2",
        12: "#33F2B5",
        13: "#1F7A3F",
        14: "#164626",
        15: "#111D15",
        16: "#CFCF1C",
        17: "#A0A019",
        18: "#4F4F16",
        19: "#E4B633",
        20: "#DF821F",
        21: "#874C0E",
        22: "#7C0E87",
        23: "#DE95E5",
        24: "#BDBDBD",
        25: "#3C6F8A",
        26: "#BB8ED5",
        27: "#4E2F61",
        28: "#7FEEFF",
        29: "#7FFFFF"
    }
    for value in sorted(grid['district'].unique()):
        df = grid[grid.district == value]
        plt.scatter(df['x']+np.random.normal(0, 0.1), df['y'] +
                    np.random.normal(0, 0.1), s=df['count'], c=district_colors[value], alpha=0.9, label=value)

    axes = plt.gca()
    axes.set_xlim([-1, 11])
    axes.set_ylim([-1, 11])
    axes.legend(loc='best', frameon=False)
    plt.show()

    print(grid.district.value_counts())

    for value in sorted(grid['district'].unique()):
        df = grid[grid.district == value]
        plt.scatter(df['x'], df['y'], s=df['count'],
                    c=district_colors[value], alpha=0.9, label=value)

        axes = plt.gca()
        axes.set_xlim([-1, 11])
        axes.set_ylim([-1, 11])
        axes.legend(loc='best', frameon=False)
        plt.show()


# Main function


def main():
    cyclic_tour()


def mp_to_grid(data):
    result = np.empty((349, 2))
    i = 0
    for sample in data:
        x = math.floor(sample/10)
        y = sample % 10
        result[i, 0] = x
        result[i, 1] = y
        i = i+1
    return result


if __name__ == "__main__":
    main()
