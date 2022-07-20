import glob

from absl import app
from absl import flags

from dataio import save_obj
from lib.nglod.lib.torchgp import load_obj, normalize

flags.DEFINE_string('shape_glob',
                    '*',
                    'Glob for all the shapes that should be normalized')
FLAGS = flags.FLAGS


def main(_):
    all_shape_paths = glob.glob(FLAGS.shape_glob)
    print(all_shape_paths)
    for shape_path in all_shape_paths:
        V, F = load_obj(shape_path)
        V, F = normalize(V, F)
        save_obj(shape_path.split('.')[0] + '_normalized.obj', V, F)


if __name__ == '__main__':
    app.run(main)
