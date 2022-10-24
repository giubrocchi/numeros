from sklearn.neural_network import MLPClassifier
import re

def open_file(name):
  file = open(name, 'r')
  file.readline()
  file.readline()
  file.readline()
  file.readline()

  file_content = file.readlines()

  for i in range(0, len(file_content)):
    file_content[i] = int(re.findall(r'\d+', file_content[i])[0])/255

  return file_content


file_names = ['./Numeros/1.pgm', './Numeros/1-2.pgm', './Numeros/1-3.pgm', './Numeros/1-4.pgm', './Numeros/1-5.pgm', './Numeros/1-6.pgm', 
              './Numeros/2.pgm', './Numeros/2-2.pgm', './Numeros/2-3.pgm', './Numeros/2-4.pgm', './Numeros/2-5.pgm', './Numeros/2-6.pgm',
              './Numeros/3.pgm', './Numeros/3-2.pgm', './Numeros/3-3.pgm', './Numeros/3-4.pgm', './Numeros/3-5.pgm', './Numeros/3-6.pgm',
              './Numeros/4.pgm', './Numeros/4-2.pgm', './Numeros/4-3.pgm', './Numeros/4-4.pgm', './Numeros/4-5.pgm', './Numeros/4-6.pgm',
              './Numeros/5.pgm', './Numeros/5-2.pgm', './Numeros/5-3.pgm', './Numeros/5-4.pgm', './Numeros/5-5.pgm', './Numeros/5-6.pgm',
              './Numeros/6.pgm', './Numeros/6-2.pgm', './Numeros/6-3.pgm', './Numeros/6-4.pgm', './Numeros/6-5.pgm', './Numeros/6-6.pgm',
              './Numeros/7.pgm', './Numeros/7-2.pgm', './Numeros/7-3.pgm', './Numeros/7-4.pgm', './Numeros/7-5.pgm', './Numeros/7-6.pgm',
              './Numeros/8.pgm', './Numeros/8-2.pgm', './Numeros/8-3.pgm', './Numeros/8-4.pgm', './Numeros/8-5.pgm', './Numeros/8-6.pgm',
              './Numeros/9.pgm', './Numeros/9-2.pgm', './Numeros/9-3.pgm', './Numeros/9-4.pgm', './Numeros/9-5.pgm', './Numeros/9-6.pgm',
              './Numeros/0.pgm', './Numeros/0-2.pgm', './Numeros/0-3.pgm', './Numeros/0-4.pgm', './Numeros/0-5.pgm', './Numeros/0-6.pgm']

expected_numbers = [1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6, 6, 6,
                    7, 7, 7, 7, 7, 7,
                    8, 8, 8, 8, 8, 8,
                    9, 9, 9, 9, 9, 9,
                    0, 0, 0, 0, 0, 0]
numbers = []
for name in file_names:
  numbers.append(open_file(name))

net = MLPClassifier(activation='logistic', solver='lbfgs', max_iter=10000000, hidden_layer_sizes=(50,10))

for i in range(0, 10):
  net.fit(numbers, expected_numbers)

tests = ['./Numeros/1-teste.pgm', './Numeros/2-teste.pgm',
        './Numeros/3-teste.pgm', './Numeros/4-teste.pgm',
        './Numeros/5-teste.pgm', './Numeros/6-teste.pgm',
        './Numeros/7-teste.pgm', './Numeros/8-teste.pgm',
        './Numeros/9-teste.pgm', './Numeros/0-teste.pgm']

for test in tests:
  print(net.predict([open_file(test)]))
