from .base import BaseModel


class ModelRegistry:
    """
    模型注册表 / Central model registry.

    Usage:
        from opthtools import registry
        registry.list_models()
        model = registry.load("myopic_maculopathy_resnet18", checkpoint_path="...")
        result = model.predict(image)
    """

    def __init__(self):
        self._models = {}

    def register(self, cls):
        """Register a model class. Use as decorator: @registry.register"""
        if not cls.name:
            raise ValueError(f"Model class {cls.__name__} must define a 'name' attribute.")
        if cls.name in self._models:
            raise ValueError(f"Model '{cls.name}' is already registered.")
        self._models[cls.name] = cls
        return cls

    def list_models(self):
        """列出所有已注册模型 / List all registered models with metadata."""
        result = []
        for name, cls in self._models.items():
            result.append({
                "name": cls.name,
                "display_name": cls.display_name,
                "task": cls.task,
                "modality": cls.modality,
                "description": cls.description,
            })
        return result

    def get_model_class(self, name):
        """Get a registered model class by name."""
        if name not in self._models:
            available = list(self._models.keys())
            raise KeyError(f"Model '{name}' not found. Available: {available}")
        return self._models[name]

    def load(self, name, checkpoint_path, device=None, **kwargs):
        """
        加载模型 / Instantiate a registered model.

        Args:
            name: Model registry name.
            checkpoint_path: Path to checkpoint file.
            device: "cuda", "mps", "cpu", or None for auto.

        Returns:
            Loaded model instance ready for prediction.
        """
        cls = self.get_model_class(name)
        return cls(checkpoint_path=checkpoint_path, device=device, **kwargs)


# Global singleton
registry = ModelRegistry()
