from kegnet.classifier.train import main as train_student
from kegnet.generator.train import main as train_generator


def main():
    dataset = 'mnist'
    classifier = '../pretrained/{}.pth.tar'.format(dataset)
    index = 0
    path_out = '../out/{}'.format(dataset)

    # gen_path = '../out/mnist/generator-200.pth.tar'
    gen_path = train_generator(dataset, classifier, index, path_out)

    data_dist = 'kegnet'
    student_index = 1
    generators = [gen_path]
    train_student(dataset, data_dist, index, path_out,
                  train=True,
                  teacher=classifier,
                  generators=generators,
                  option=student_index)


if __name__ == '__main__':
    main()
