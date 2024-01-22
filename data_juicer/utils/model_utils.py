import fnmatch
import os
from functools import partial

import wget
from loguru import logger

from .cache_utils import DATA_JUICER_MODELS_CACHE as DJMC

MODEL_ZOO = {}

# Default cached models links for downloading
MODEL_LINKS = 'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/' \
                'data_juicer/models/'

# Backup cached models links for downloading
BACKUP_MODEL_LINKS = {
    # language identification model from fasttext
    'lid.176.bin':
    'https://dl.fbaipublicfiles.com/fasttext/supervised-models/',

    # tokenizer and language model for English from sentencepiece and KenLM
    '*.sp.model':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',
    '*.arpa.bin':
    'https://huggingface.co/edugp/kenlm/resolve/main/wikipedia/',

    # sentence split model from nltk punkt
    'punkt.*.pickle':
    'https://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/'
    'data_juicer/models/',
}


def get_backup_model_link(model_name):
    for pattern, url in BACKUP_MODEL_LINKS.items():
        if fnmatch.fnmatch(model_name, pattern):
            return url
    return None


def check_model(model_name, force=False):
    """
    Check whether a model exists in MODEL_PATH. If exists, return its full path
    Else, download it from cached models links.

    :param model_name: a specified model name
    :param args: optional extra args of model.
    :param force: Whether to download model forcefully or not, Sometimes
        the model file maybe incomplete for some reason, so need to
        download again forcefully.
    """
    # check for local model
    if os.path.exists(model_name):
        return model_name

    if not os.path.exists(DJMC):
        os.makedirs(DJMC)

    # check if the specified model exists. If it does not exist, download it
    cached_model_path = os.path.join(DJMC, model_name)
    if force:
        if os.path.exists(cached_model_path):
            os.remove(cached_model_path)
            logger.info(f'Model [{model_name}] exists and will be replaced. '
                        'Force to downloading...')
        else:
            logger.info(f'Model [{model_name}] is not found. Downloading...')

        try:
            model_link = os.path.join(MODEL_LINKS, model_name)
            wget.download(model_link, cached_model_path, bar=None)
        except:  # noqa: E722
            try:
                backup_model_link = os.path.join(
                    get_backup_model_link(model_name), model_name)
                wget.download(backup_model_link, cached_model_path, bar=None)
            except:  # noqa: E722
                logger.error(
                    f'Downloading model [{model_name}] error. '
                    f'Please retry later or download it into {DJMC} '
                    f'manually from {model_link} or {backup_model_link} ')
                exit(1)
    return cached_model_path


def prepare_fasttext_model(model_name='lid.176.bin'):
    """
    Prepare and load a fasttext model.

    :param model_name: input model name
    :return: model instance.
    """
    import fasttext

    logger.info('Loading fasttext language identification model...')
    try:
        ft_model = fasttext.load_model(check_model(model_name))
    except:  # noqa: E722
        ft_model = fasttext.load_model(check_model(model_name, force=True))
    return ft_model


def prepare_sentencepiece_model(model_name='en.sp.model'):
    """
    Prepare and load a sentencepiece model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """
    import sentencepiece

    logger.info('Loading sentencepiece model...')
    sentencepiece_model = sentencepiece.SentencePieceProcessor()
    try:
        sentencepiece_model.load(check_model(model_name))
    except:  # noqa: E722
        sentencepiece_model.load(check_model(model_name, force=True))
    return sentencepiece_model


def prepare_kenlm_model(model_name='en.arpa.bin'):
    """
    Prepare and load a kenlm model.

    :param model_name: input model name in formatting syntax.
    :param lang: language to render model name
    :return: model instance.
    """
    import kenlm

    logger.info('Loading kenlm language model...')
    try:
        kenlm_model = kenlm.Model(check_model(model_name))
    except:  # noqa: E722
        kenlm_model = kenlm.Model(check_model(model_name, force=True))
    return kenlm_model


def prepare_nltk_model(model_name='punkt.english.pickle'):
    """
    Prepare and load a nltk punkt model.

    :param model_name: input model name in formatting syntax
    :param lang: language to render model name
    :return: model instance.
    """

    valid_langs = ('english', 'french', 'portuguese', 'spanish')
    name_parts = model_name.split('.')
    if name_parts > 1:
        lang = name_parts[1]
        assert (lang in valid_langs(
        )), 'language must be one of the following: {}'.format(valid_langs)

    from nltk.data import load

    logger.info('Loading nltk punkt split model...')
    try:
        nltk_model = load(check_model(model_name))
    except:  # noqa: E722
        nltk_model = load(check_model(model_nameforce=True))
    return nltk_model


def prepare_huggingface_model(model_name,
                              with_model=True,
                              trust_remote_code=False):
    import transformers
    from transformers import (AutoConfig, AutoImageProcessor, AutoProcessor,
                              AutoTokenizer)
    from transformers.models.auto.image_processing_auto import \
        IMAGE_PROCESSOR_MAPPING_NAMES
    from transformers.models.auto.processing_auto import \
        PROCESSOR_MAPPING_NAMES
    from transformers.models.auto.tokenization_auto import \
        TOKENIZER_MAPPING_NAMES

    config = AutoConfig.from_pretrained(model_name)
    # TODO: What happens when there are more than one?
    arch = config.architectures[0]
    model_class = getattr(transformers, arch)
    model_type = config.model_type
    if model_type in PROCESSOR_MAPPING_NAMES:
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    elif model_type in IMAGE_PROCESSOR_MAPPING_NAMES:
        processor = AutoImageProcessor.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    elif model_type in TOKENIZER_MAPPING_NAMES:
        processor = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
    else:
        processor = None

    if with_model:
        model = model_class.from_pretrained(model_name)
    return (model, processor) if with_model else processor


def prepare_diversity_model(model_name='en_core_web_md-3.5.0.zip'):
    """
    Prepare diversity model for specific language.

    :param model_name: the model name to be loaded.
    :param lang: language of diversity model. Should be one of ["zh",
        "en"]
    :return: corresponding diversity model
    """
    import spacy

    lang = model_name.split('_')[0]
    assert lang in ('zh', 'en'), 'Diversity only support zh and en'
    logger.info(f'Loading spacy model [{model_name}]...')

    # decompress the compressed model if it's not decompressed
    def decompress_model(compressed_model_path):
        decompressed_model_path = compressed_model_path.replace('.zip', '')
        if os.path.exists(decompressed_model_path) and os.path.isdir(
                decompressed_model_path):
            return decompressed_model_path
        import zipfile

        with zipfile.ZipFile(compressed_model_path) as zf:
            zf.extractall(DJMC)
        return decompressed_model_path

    try:
        diversity_model = spacy.load(decompress_model(check_model(model_name)))
    except:  # noqa: E722
        diversity_model = spacy.load(
            decompress_model(check_model(model_name, force=True)))
    return diversity_model


MODEL_FUNCTION_MAPPING = {
    'fasttext': prepare_fasttext_model,
    'sentencepiece': prepare_sentencepiece_model,
    'kenlm': prepare_kenlm_model,
    'nltk': prepare_nltk_model,
    'huggingface': prepare_huggingface_model,
    'spacy': prepare_diversity_model,
}


def prepare_model(model_type, **model_kwargs):
    assert (model_type in MODEL_FUNCTION_MAPPING.keys()
            ), 'model_type must be one of the following: {}'.format(
                list(MODEL_FUNCTION_MAPPING.keys()))
    global MODEL_ZOO
    model_func = MODEL_FUNCTION_MAPPING[model_type]
    model_key = partial(model_func, **model_kwargs)
    # always instantiate once for possible caching
    model_objects = model_key()
    MODEL_ZOO[model_key] = model_objects
    return model_key


def get_model(model_key):
    global MODEL_ZOO
    if model_key is None:
        logger.warning('Please specify model_key to get models')
        return None
    if model_key not in MODEL_ZOO:
        MODEL_ZOO[model_key] = model_key()
    return MODEL_ZOO[model_key]
