"""Create a fresh local config with random infra credentials; never overwrite or call providers."""
import argparse
from pathlib import Path
import secrets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('.env'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    template = (root / '.env.example').read_text(encoding='utf8')
    password = secrets.token_hex(24)
    values = {'MYSQL_ROOT_PASSWORD': secrets.token_hex(24), 'MYSQL_PASSWORD': password,
              'MINIO_ROOT_USER': 'ec' + secrets.token_hex(8), 'MINIO_ROOT_PASSWORD': secrets.token_hex(24),
              'DATABASE_URL': f'mysql+asyncmy://ecomcare:{password}@127.0.0.1:13307/mewhelp',
              'TEST_DATABASE_URL': f'mysql+asyncmy://ecomcare:{password}@127.0.0.1:13307/mewhelp_test'}
    text = '\n'.join(key + '=' + values[key] if '=' in line and (key := line.split('=', 1)[0]) in values else line
                     for line in template.splitlines()) + '\n'
    with args.output.open('x', encoding='utf8') as stream:
        stream.write(text)
    try:
        args.output.chmod(0o600)
    except OSError:
        pass
    print('Created private local config; provider keys remain empty. Existing config was not overwritten.')


if __name__ == '__main__':
    main()
