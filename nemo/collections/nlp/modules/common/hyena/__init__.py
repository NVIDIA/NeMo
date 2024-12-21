try:
    from nemo.collections.nlp.modules.common.hyena.hyena import HyenaOperator

except ImportError:

    HAVE_TE = False
