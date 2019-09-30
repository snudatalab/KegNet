"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import os

from kegnet.classifier.train import main as train_student
from kegnet.generator.train import main as train_generator


def main():
    dataset = 'mnist'
    classifier = '../pretrained/{}.pth.tar'.format(dataset)
    path_out = '../out/{}'.format(dataset)
    index = 0

    gen_path = '../out/mnist/generator-200.pth.tar'
    if not os.path.exists(gen_path):
        gen_path = train_generator(dataset, classifier, path_out, index)

    data_dist = 'kegnet'
    option = 1
    generators = [gen_path]
    train_student(dataset, data_dist, path_out, index,
                  load=classifier,
                  generators=generators,
                  option=option)


if __name__ == '__main__':
    main()
