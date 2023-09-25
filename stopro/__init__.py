# -*- coding: utf-8 -*-
"""
Initializes this package with metadata.
"""

from .metadata import (
        __version__,
        __author__,
        __copyright__,
        __credits__,
        __license__,
        __maintainer__,
        __email__,
        __status__,
    )

from .stopro import (
        wiener,
        ornstein_uhlenbeck,
        exponential_ornstein_uhlenbeck,
        gillespie_replicator,
        kimura_replicator,
        white_replicator,
        colored_replicator,
        geometric_brownian_motion,
        colored_geometric_brownian_motion,
        integrated_ornstein_uhlenbeck,
        moran,
        competitive_lotka_volterra,

    )
