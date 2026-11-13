"""config-validation: stop the run before any GPU time is spent when a setting is out of bounds."""
from kaos_lib import step

from app.manager import Manager


def main() -> None:
    ctx = step.start()
    Manager().run(ctx.params, ctx.app, ctx.experiment)
    step.result(checked=True)


if __name__ == "__main__":
    main()
