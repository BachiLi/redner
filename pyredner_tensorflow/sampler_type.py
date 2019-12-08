import redner
import tensorflow as tf

class SamplerType:
    def __init__(self):
        self.independent = redner.SamplerType.independent
        self.sobol = redner.SamplerType.sobol

sampler_type = SamplerType()

class RednerSamplerType:
    __samplertypes = [
        redner.SamplerType.independent,
        redner.SamplerType.sobol
    ]

    @staticmethod
    def asTensor(samplertype: redner.SamplerType) -> tf.Tensor:
        assert isinstance(samplertype, redner.SamplerType)

        for i in range(len(RednerSamplerType.__samplertypes)):
            if RednerSamplerType.__samplertypes[i] == samplertype:
                return tf.constant(i)


    @staticmethod
    def asSamplerType(index: tf.Tensor) -> redner.SamplerType:
        try:
            samplertype = RednerSamplerType.__samplertypes[index]
        except IndexError:
            print(f'{index} is out of range: [0, {len(RednerSamplerType.__samplertypes)})')
            import sys
            sys.exit()
        else:
            return samplertype
