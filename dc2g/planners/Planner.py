import os
import matplotlib.pyplot as plt

class Planner:
    def __init__(self, name, traversable_colors, goal_color, room_or_object_goal, env_to_coor, env_next_coords, env_to_grid, env_grid_resolution):
        self.name = name
        self.traversable_colors = traversable_colors
        self.goal_color = goal_color
        self.room_or_object_goal = room_or_object_goal

        self.step_number = 0
        self.project_path = os.path.dirname(os.path.realpath(__file__))+'/../..'
        self.individual_figure_path = self.project_path+'/dc2g/results'

    def step(self, obs):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def setup_plots(self, make_individual_figures, plot_panels, save_panel_figures, save_individual_figures):
        self.make_individual_figures = make_individual_figures
        self.plot_panels = plot_panels
        self.save_panel_figures = save_panel_figures
        self.save_individual_figures = save_individual_figures
        
        if make_individual_figures:
            fig = plt.figure('C2G', figsize=(12, 12))
            plt.axis('off')
            fig = plt.figure('Observation', figsize=(12, 12))
            plt.axis('off')
            fig = plt.figure('Environment', figsize=(12, 12))
            plt.axis('off')
        if plot_panels:
            fig = plt.figure('DC2G', figsize=(12, 5))
            ax = fig.add_subplot(131)
            ax.set_axis_off()
            ax.set_title('2D Partial Semantic Map')
            ax = fig.add_subplot(132)
            ax.set_axis_off()
            ax.set_title('Predicted Cost-to-Go')
            ax = fig.add_subplot(133)
            ax.set_axis_off()
            ax.set_title('Planned Path')
            ax.set_xlabel("Red: Path\nBlue: Frontiers\nGray: Reachable Pts\nGreen: Reachable Cells to Push Frontier")
            plt.xticks(visible=False)
            plt.yticks(visible=False)

            # ax = fig.add_subplot(231)
            # ax.set_axis_off()
            # ax.set_title('World Semantic Map')
            # ax = fig.add_subplot(232)
            # ax.set_axis_off()
            # ax.set_title('True Cost-to-Go')
            # ax = fig.add_subplot(233)
            # ax.set_axis_off()
            # ax.set_title('Agent in Semantic Map')
            # total_world_array = plt.imread(world_image_filename)
            # world_id = world_image_filename.split('/')[-1].split('.')[0] # e.g. world00asdasdf
            # world_filetype = world_image_filename.split('/')[-1].split('.')[-1] # e.g. png
            # c2g_image_filename = '/home/mfe/code/dc2g/training_data/{dataset}/full_c2g/test/{world}{target}.{filetype}'.format(filetype=world_filetype, target=target, dataset=dataset, world=world_id)
            # total_c2g_array = plt.imread(c2g_image_filename)
            # plt.figure("DC2G")
            # plt.subplot(231)
            # plt.imshow(total_world_array)
            # plt.figure("DC2G")
            # plt.subplot(232)
            # plt.imshow(total_c2g_array)

    def plot(self, full_semantic_array):
        if self.plot_panels:
            plt.figure("DC2G")
            plt.suptitle('Step:' + str(self.step_number), fontsize=20)
            # try:
            #     render = ENVIRONMENT.render(mode=render_mode, show_trajectory=True) # TODO: add support for show_trajectory to House3D
            # except:
            # render = np.zeros((100,100))
            # plt.subplot(233)
            # plt.imshow(render)
            # if make_individual_figures:
            #     plt.figure("Environment")
            #     plt.imshow(render)

            plt.figure("DC2G")
            plt.subplot(131)
            plt.imshow(full_semantic_array)

            if self.save_panel_figures:
                plt.figure("DC2G")
                plt.savefig("{individual_figure_path}/panels/step_{step_num}.png".format(individual_figure_path=self.individual_figure_path, step_num=str(self.step_number).zfill(3)))
            if self.save_individual_figures:
                # plt.imsave("{individual_figure_path}/results/environment/step_{step_num}.png".format(individual_figure_path=self.individual_figure_path, step_num=str(self.step_number).zfill(3)), render)
                plt.imsave("{individual_figure_path}/observation/step_{step_num}.png".format(individual_figure_path=self.individual_figure_path, step_num=str(self.step_number).zfill(3)), full_semantic_array)
            plt.pause(0.01)