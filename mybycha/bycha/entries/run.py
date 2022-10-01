from bycha.tasks import create_task
from bycha.utils.runtime import build_env
from bycha.entries.util import parse_config


def main():
    confs = parse_config()
    if 'env' in confs:
        build_env(confs['task'], **confs['env'])
    task = create_task(confs.pop('task'))
    task.build()
    task.run()


if __name__ == '__main__':
    main()
