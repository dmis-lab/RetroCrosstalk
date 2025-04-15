import logging


def setup_logger(fname=None, silent=False):
    if fname is None:
        logging.basicConfig(
            level=logging.INFO if not silent else logging.CRITICAL,
            format='%(name)-12s: %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filemode='w'
        )
    else:
        logging.basicConfig(
            level=logging.INFO if not silent else logging.CRITICAL,
            format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d %H:%M',
            filename=fname,
            filemode='w'
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
        
# def setup_logger():
#     logging.basicConfig(
#         level=logging.INFO,  # 명시적으로 DEBUG 레벨 설정
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
