try:
    from .app import main
except ImportError:
    from effort_estimator_package.app import main


if __name__ == "__main__":
    main()