from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(confusion_mat, title: str = 'Confusion Matrix', classes=None):
	if classes is None:
		classes = [-1, 0, 1]
	plt.figure()

	ind_array = np.arange(len(classes))
	x, y = np.meshgrid(ind_array, ind_array)
	for x_val, y_val in zip(x.flatten(), y.flatten()):
		c = confusion_mat[y_val][x_val]
		plt.text(x_val, y_val, "%0.2f" % (c,), color='black', fontsize=15, va='center', ha='center')

	plt.imshow(confusion_mat, interpolation='nearest', cmap='Wistia')
	plt.title(title, fontdict={'fontsize': 18})
	plt.colorbar(extend='both')
	x_locations = np.array(range(len(classes)))
	plt.xticks(x_locations, classes, rotation=90)
	plt.yticks(x_locations, classes)
	plt.ylabel('Actual sentiment', fontdict={'fontsize': 12})
	plt.xlabel('Predict sentiment', fontdict={'fontsize': 12})

	# offset the tick
	tick_marks = np.array(range(len(classes))) + 0.5
	plt.gca().set_xticks(tick_marks, minor=True)
	plt.gca().set_yticks(tick_marks, minor=True)
	plt.gca().xaxis.set_ticks_position('none')
	plt.gca().yaxis.set_ticks_position('none')
	plt.grid(True, which='minor', linestyle='-')
	plt.gcf().subplots_adjust(bottom=0.15)

	# show confusion matrix
	plt.show()
	plt.close()


