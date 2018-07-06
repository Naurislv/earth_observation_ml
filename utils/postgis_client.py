"""Postgist client to get necesarry data for training from DB."""

# Dependency imports
import psycopg2
import psycopg2.extras
import pandas as pd

class GisDB():
    """Connect to PostGIS DB. Implementations of ready to use
    queries to get necesarry data from this DB."""

    def __init__(self):

        # create connection to ices database
        dbname = 'geo'
        host = 'lvmgeodb.cgjy0yyob75n.eu-central-1.rds.amazonaws.com'
        user = 'nauris'
        password = '8r3tFXUntQeY'

        conn = psycopg2.connect(f"dbname={dbname} host={host} user={user} password={password}")
        self.cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        self._class_maps = None
        self._nb_class = None
        self._data = None

    @property
    def class_maps(self):
        """Return class name mappings to IDs."""

        if self._class_maps is None:

            class_maps = self._get_class_maps()
            self._class_maps = class_maps

        return self._class_maps

    @property
    def class_maps_inv(self):
        """Return ID mappings to class names."""

        if self._class_maps is None:

            class_maps = self._get_class_maps()
            self._class_maps = class_maps

        return {v: k for k, v in self._class_maps.items()}

    @property
    def nb_class(self):
        """Return each main class distribution over dataset."""

        if self._nb_class is None:
            self.cur.execute("SELECT count(m.s10) as skaits, c.name as suga FROM "
                             "lvm_data_all as m, lvm_tree_classes as c "
                             "WHERE m.s10=c.classid "
                             "GROUP BY c.name "
                             "ORDER BY skaits DESC;")

            nb_class = {}

            for row in self.cur:
                nb_class[row[1].replace('\t', '')] = row[0]

            self._nb_class = nb_class

        return self._nb_class

    @property
    def data(self):
        """Return all dataset as pandas dataframe."""

        if self._data is None:
            self.cur.execute("SELECT objectid::int, "
                             "s10::int,      s11::int,      s12::int, "
                             "k10/10::float, k11/10::float, k12/10::float, "
                             "h10::float,    h11::float,    h12::float, "
                             "a10::int,      a11::int,      a12::int, "
                             "v10::float,    v11::float,    v12::float "
                             "FROM lvm_data_all "
                             "WHERE s10 IS NOT NULL;")

            data = []

            for row in self.cur:
                data.append(list(row))

            self._data = pd.DataFrame(
                data,
                columns=[
                    'objectid',
                    's1', 's2', 's3',
                    'k1', 'k2', 'k3',
                    'h1', 'h2', 'h3',
                    'a1', 'a2', 'a3',
                    'v1', 'v2', 'v3',
                    ]
            )

        return self._data

    def _get_class_maps(self):
        """Query class mapping data from DB."""
        self.cur.execute("SELECT c.classid::int, c.name::text "
                         "FROM lvm_tree_classes as c ")

        class_maps = {}

        for row in self.cur:
            class_id = row[0]
            class_name = row[1]
            class_maps[class_id] = class_name.replace('\t', '')

        return class_maps
