from manafaln.adapters.monailabel.apps import RadiologyApp

# Workaround for MONAILabel get_class_names
RadiologyApp.__module__ = __name__
