class Configurations:
    """
    Class to store simulation configuration.

    Attibutes
    ---------
    flag_preLoadedgeometry = 1 # if the program need to input data from folder set true = 1
    meshdir = "small_mesh" # Place to preload
    flag_selfWeight = 0 # Choose if the selfweight is applied in problem (Yes = 1)
    flag_planeStressorStrain = 0 # Plane stress problem = 1; Plane strain problem = 0
    
    Methods
    -------
    config_rectangular: using lenght, height and number of nodes provides for each edge
    """

    def __init__(self, dict_environment):
        self.flag_preLoadedgeometry = dict_environment["flag_preLoadedgeometry"]
        self.meshdir = dict_environment["meshdir"]
        self.flag_selfWeight = dict_environment["flag_selfWeight"]
        self.flag_planeStressorStrain = dict_environment["flag_planeStressorStrain"]

    def config_rectangular(self, dict_environment):
        self.rect_length = dict_environment["rect_length"]
        self.rect_heigth = dict_environment["rect_heigth"]
        self.thick = dict_environment["thick"]
        self.NNodes_edge = dict_environment["NNodes_long_edge"]

    def config_pointload(self, dict_environment):
        self.x_load = dict_environment["load_x"]
        self.y_load = dict_environment["load_y"]
        self.direction = dict_environment["load_direction"]
        self.magnitude = dict_environment["load_magnitude"]

    def config_material(self, dict_environment):
        self.young_modulus = dict_environment["material_young"]
        self.poisson_ratio = dict_environment["material_poisson"]
        self.density = dict_environment["material_density"]

    def update_parameters(self, dict_update):
        self.config_rectangular(dict_update)
        self.config_pointload(dict_update)
        self.config_material(dict_update)
