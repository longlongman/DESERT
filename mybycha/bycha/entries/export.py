from bycha.tasks import create_task
from bycha.utils.runtime import build_env
from bycha.entries.util import parse_config


def main():
    confs = parse_config()
    if 'env' in confs:
        build_env(confs['task'], **confs['env'])
    export_conf = confs.pop('export')
    task = create_task(confs.pop('task'))
    task.build()
    path = export_conf.pop("path")
    task.export(path, **export_conf)


if __name__ == '__main__':
    main()
