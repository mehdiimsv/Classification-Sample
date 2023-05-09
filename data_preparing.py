import os
import csv
import argparse


def main(args):
	name = args.name
	data_path = os.path.join(args.data_path, name)
	full_path = os.path.abspath(data_path)
	folder_names = os.listdir(full_path)

	with open(f'{name}.csv', 'a') as file:
		writer = csv.writer(file)
		for folder_name in folder_names:
			folder_path = os.path.join(full_path, folder_name)
			img_names = os.listdir(folder_path)
			folder_labels = len(img_names) * [folder_name]

			for name, label in zip(img_names, folder_labels):
				writer.writerow([name, label])


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='All needed parameters')
	parser.add_argument('--data_path', required = True, help='dataset path')
	parser.add_argument('--name', required = True, help='train or test')

	arguments = parser.parse_args()

	main(arguments)
