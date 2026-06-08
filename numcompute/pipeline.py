from typing import Any, Sequence

class Pipeline:
    def __init__(self, steps: Sequence[tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("steps must contain at least one name and object pair.")
        
        self.steps = list(steps)
        self.named_steps = {}

        for name, step in self.steps:
            if name in self.named_steps:
                raise ValueError(f"duplicate step name: {name}")
            self.named_steps[name] = step
            
            if not hasattr(step, 'fit'):
                raise ValueError("step must implement 'fit'.")
        
        for name, step in self.steps[:-1]:
            if not hasattr(step, 'transform'):
                raise ValueError("non-final step must implement 'transform'.")
        last_name, last_step = self.steps[-1]
        if not hasattr(last_step, 'transform') and not hasattr(last_step, 'predict'):
            raise ValueError("final step must implement 'transform' or 'predict'.")

    def fit(self, X: Any, y: Any = None) -> "Pipeline":
        x_curr = X
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1
            if is_last:
                if y is None: 
                    step.fit(x_curr)
                else: 
                    step.fit(x_curr, y)
            else:
                if hasattr(step, "fit_transform"):
                    x_curr = step.fit_transform(x_curr)
                else:
                    step.fit(x_curr)
                    x_curr = step.transform(x_curr)
        return self

    def partial_fit(self, X: Any, y: Any = None) -> "Pipeline":
        """
        Incremental fit for streaming data.
        """
        x_curr = X
        for i, (name, step) in enumerate(self.steps):
            is_last = i == len(self.steps) - 1

            if hasattr(step, "partial_fit"):
                if is_last and y is not None:
                    step.partial_fit(x_curr, y)
                else:
                    step.partial_fit(x_curr)
            elif hasattr(step, "fit"):
                if is_last and y is not None:
                    step.fit(x_curr, y)
                else:
                    step.fit(x_curr)

            if not is_last:
                x_curr = step.transform(x_curr)
                
        return self

    def transform(self, X: Any) -> Any:
        x_curr = X
        for _, step in self.steps:
            if hasattr(step, "transform"):
                x_curr = step.transform(x_curr)
        return x_curr

    def predict(self, X: Any) -> Any:
        x_curr = X
        for name, step in self.steps[:-1]:
            x_curr = step.transform(x_curr)
        
        last_step = self.steps[-1][1]
        if not hasattr(last_step, 'predict'):
            raise ValueError(f"Final step '{self.steps[-1][0]}' does not implement 'predict'.")
        return last_step.predict(x_curr)

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)

__all__ = ["Pipeline"]
