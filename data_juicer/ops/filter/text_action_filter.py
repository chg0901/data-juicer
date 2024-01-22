from data_juicer.utils.constant import Fields, StatsKeys
from data_juicer.utils.mm_utils import remove_special_tokens
from data_juicer.utils.model_utils import get_model, prepare_model

from ..base_op import OPERATORS, Filter

OP_NAME = 'text_action_filter'


@OPERATORS.register_module(OP_NAME)
class TextActionFilter(Filter):
    """
    Filter to keep texts those contain actions in the text.
    """

    def __init__(self,
                 spacy_model = 'en_core_web_md-3.5.0.zip',
                 min_action_num: int = 1,
                 *args,
                 **kwargs):
        """
        Initialization method.

        :param lang: language of the text in the samples. 'en' for detection of
            actions in English and 'zh' for detection of actions in Chinese.
        :param mini_action_num: The min action number in the filtering. samples
            will be filtered if their action number in the text is below this
            parameter.
        """
        super().__init__(*args, **kwargs)

        self.model_key = prepare_model(model_type='spacy', model_name=spacy_model)
        self.action_poss = ['VERB']
        self.action_tags = ['VV', 'VB', 'VBP', 'VBZ', 'VBD', 'VBG', 'VBN']
        self.min_action_num = min_action_num

    def compute_stats(self, sample, context=False):
        # check if it's computed already
        if StatsKeys.num_action in sample[Fields.stats]:
            return sample

        text = remove_special_tokens(sample[self.text_key])

        # process text via spacy and count the actions in text
        model = get_model(self.model_key)
        doc = model(text)
        num_action = 0
        for token in doc:
            if token.pos_ in self.action_poss \
             and token.tag_ in self.action_tags:
                num_action += 1
        sample[Fields.stats][StatsKeys.num_action] = num_action

        return sample

    def process(self, sample):
        num_action = sample[Fields.stats][StatsKeys.num_action]
        if self.min_action_num <= num_action:
            return True
        else:
            return False
