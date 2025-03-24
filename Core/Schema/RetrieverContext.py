from dataclasses import dataclass, asdict, field

@dataclass
class RetrieverContext:
    context: dict = field(default_factory=dict)
    def register_context(self, key, value):
       self.context[key] = value
       
    @property
    def as_dict(self):
        return self.context
    
    @property
    def config(self):
        return self.context["config"]
    
    @property
    def llm(self):
        return self.context["llm"]