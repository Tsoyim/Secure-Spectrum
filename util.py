# get the saving path
import os
import datetime

class Util:
    def __init__(self, n_CUE, n_VUE, n_Eve, Eveknow, algrothim):
        if Eveknow == 1:
            label = 'know'
        else:
            label = 'unknow'
        # current_time = datetime.datetime.now()
        # self.dir_label = 'experiment_result/{}/Eve_{}/'.format(algrothim, label)
        self.dir_label = 'experiment_result/{}/'.format(algrothim, label)
        self.current_directory = os.getcwd()

    def get_model_path(self,model_name):
        dir_path = os.path.join(self.current_directory, self.dir_label, 'model')
        path = os.path.join(dir_path, model_name)
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)
        return dir_path, path

    def get_train_result_path(self,result_type, result_name):
        dir_path = os.path.join(self.current_directory, self.dir_label, 'train_result')
        data_path = ''
        if result_type == 'reward':
            path = os.path.join(dir_path, '{}.png'.format(result_name))
            data_path = os.path.join(dir_path, '{}.npy'.format(result_name))
        elif result_type == 'loss':
            path = os.path.join(dir_path, 'loss.png')
            data_path = os.path.join(dir_path, 'loss.npy')
        elif result_type == 'config':
            path = os.path.join(dir_path, '{}.txt'.format(result_name))
        else:
            path = None
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)
        return dir_path, path, data_path

    def get_reward_fig_path(self):
        dir_path = os.path.join(self.current_directory, self.dir_label, 'train_result')
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)

        reward_path = os.path.join(dir_path, 'reward.png')
        ma_reward_path = os.path.join(dir_path, 'ma_reward.png')
        return dir_path, reward_path, ma_reward_path
    
    def get_fig_path(self, name):
        dir_path = os.path.join(self.current_directory, self.dir_label, 'test_result')
        path = os.path.join(dir_path, '{}.png'.format(name))
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)

        return dir_path, path
        

    def get_test_result_path(self, name):
        dir_path = os.path.join(self.current_directory, self.dir_label, 'test_result')
        path = os.path.join(dir_path, name)
        if os.path.isdir(dir_path) == False:
            os.makedirs(dir_path)

        return dir_path, path
    # def get_test_rate_array_path(self, test_episode):
    #     dir_path = os.path.join(self.current_directory, self.dir_label, 'test_result')
    #     v2v_path = os.path.join(dir_path, 'v2v_rate_{}.npz'.format(test_episode))
    #     v2i_path = os.path.join(dir_path, 'v2i_rate_{}.npz'.format(test_episode))
    #     if os.path.isdir(dir_path) == False:
    #         os.makedirs(dir_path)

    #     return dir_path, v2v_path, v2i_path

    # def get_test_probability_path(self, test_episode):
    #     dir_path = os.path.join(self.current_directory, self.dir_label, 'test_result')
    #     v2v_path = os.path.join(dir_path, 'V2V_pro_{}.png'.format(test_episode))
    #     v2i_path = os.path.join(dir_path, 'V2I_pro_{}.png'.format(test_episode))
    #     if os.path.isdir(dir_path) == False:
    #         os.makedirs(dir_path)
    #     return dir_path, v2v_path, v2i_path
    # def get_test_rate_path(self, test_episode):
    #     dir_path = os.path.join(self.current_directory, self.dir_label, 'test_result')
    #     v2v_fig_path = os.path.join(dir_path, 'V2V_sec_{}.png'.format(test_episode))
    #     v2i_fig_path = os.path.join(dir_path, 'V2I_sec_{}.png'.format(test_episode))
    #     if os.path.isdir(dir_path) == False:
    #         os.makedirs(dir_path)
    #     return dir_path, v2v_fig_path, v2i_fig_path
    
    
    def calculate_moving_average_reward(self,rewards, window_size):
        if len(rewards) <= window_size:
            return sum(rewards) / len(rewards)
        else:
            recent_rewards = rewards[-window_size:]
            return sum(recent_rewards) / window_size

# if __name__ == '__main__':
#     dir_path, path = get_model_path('agent.pth')
#     os.removedirs(dir_path)
#     print(path)
