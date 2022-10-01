import os
from bycha.tokenizers import registry, AbstractTokenizer
from bycha.utils.runtime import build_env
from bycha.entries.util import parse_config


def main():
    configs = parse_config()
    if 'env' in configs:
        env_conf = configs.pop('env')
        build_env(configs, **env_conf)
    cls = registry[configs.pop('class').lower()]
    assert issubclass(cls, AbstractTokenizer)
    os.makedirs('/'.join(configs['output_path'].split('/')[:-1]), exist_ok=True)
    data = configs.pop('data')
    cls.learn(data, **configs)


if __name__ == '__main__':
    main()
