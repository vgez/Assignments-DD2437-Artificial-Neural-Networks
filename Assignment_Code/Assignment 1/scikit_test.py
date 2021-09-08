from sklearn.neural_network import MLPClassifier
import data_generation as data

x, y = data.create_data_enc()


clf = MLPClassifier(solver='sgd', alpha=0.001,
                    hidden_layer_sizes=(3))

clf.fit(x, y)

output = clf.predict([[-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]])
print(output)
