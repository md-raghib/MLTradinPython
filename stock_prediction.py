{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyNVtXPUNvmnS9wkCI3Uh0HQ"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "twi4S_Vobhv-"
      },
      "source": [
        "# Description : this program prediscts the stock market using LSTM which is an artificial neural network\n",
        "\n",
        "import math\n",
        "import pandas_datareader as web\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, LSTM\n",
        "import matplotlib.pyplot as plt\n",
        "from datetime import datetime\n",
        "plt.style.use('fivethirtyeight')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ct9di5h7bqwa",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 419
        },
        "outputId": "c1eb8055-9b94-48de-b6bc-925b8f9a4228"
      },
      "source": [
        "#get the stock quote\n",
        "df = web.DataReader(\"AAPL\", \"av-daily\", start=datetime(2010, 1, 1),end=datetime(2021, 11, 19),api_key='YJNQH53YAREKNLE4')\n",
        "#show the data\n",
        "df"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>open</th>\n",
              "      <th>high</th>\n",
              "      <th>low</th>\n",
              "      <th>close</th>\n",
              "      <th>volume</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2010-01-04</th>\n",
              "      <td>213.430</td>\n",
              "      <td>214.500</td>\n",
              "      <td>212.3800</td>\n",
              "      <td>214.01</td>\n",
              "      <td>17633200</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2010-01-05</th>\n",
              "      <td>214.600</td>\n",
              "      <td>215.590</td>\n",
              "      <td>213.2500</td>\n",
              "      <td>214.38</td>\n",
              "      <td>21496600</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2010-01-06</th>\n",
              "      <td>214.380</td>\n",
              "      <td>215.230</td>\n",
              "      <td>210.7500</td>\n",
              "      <td>210.97</td>\n",
              "      <td>19720000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2010-01-07</th>\n",
              "      <td>211.750</td>\n",
              "      <td>212.000</td>\n",
              "      <td>209.0500</td>\n",
              "      <td>210.58</td>\n",
              "      <td>17040400</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2010-01-08</th>\n",
              "      <td>210.300</td>\n",
              "      <td>212.000</td>\n",
              "      <td>209.0600</td>\n",
              "      <td>211.98</td>\n",
              "      <td>15986100</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-15</th>\n",
              "      <td>150.370</td>\n",
              "      <td>151.880</td>\n",
              "      <td>149.4300</td>\n",
              "      <td>150.00</td>\n",
              "      <td>59222803</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-16</th>\n",
              "      <td>149.940</td>\n",
              "      <td>151.488</td>\n",
              "      <td>149.3400</td>\n",
              "      <td>151.00</td>\n",
              "      <td>59256210</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-17</th>\n",
              "      <td>150.995</td>\n",
              "      <td>155.000</td>\n",
              "      <td>150.9900</td>\n",
              "      <td>153.49</td>\n",
              "      <td>88807000</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-18</th>\n",
              "      <td>153.710</td>\n",
              "      <td>158.670</td>\n",
              "      <td>153.0500</td>\n",
              "      <td>157.87</td>\n",
              "      <td>137827673</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-19</th>\n",
              "      <td>157.650</td>\n",
              "      <td>161.020</td>\n",
              "      <td>156.5328</td>\n",
              "      <td>160.55</td>\n",
              "      <td>117305597</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>2993 rows × 5 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "               open     high       low   close     volume\n",
              "2010-01-04  213.430  214.500  212.3800  214.01   17633200\n",
              "2010-01-05  214.600  215.590  213.2500  214.38   21496600\n",
              "2010-01-06  214.380  215.230  210.7500  210.97   19720000\n",
              "2010-01-07  211.750  212.000  209.0500  210.58   17040400\n",
              "2010-01-08  210.300  212.000  209.0600  211.98   15986100\n",
              "...             ...      ...       ...     ...        ...\n",
              "2021-11-15  150.370  151.880  149.4300  150.00   59222803\n",
              "2021-11-16  149.940  151.488  149.3400  151.00   59256210\n",
              "2021-11-17  150.995  155.000  150.9900  153.49   88807000\n",
              "2021-11-18  153.710  158.670  153.0500  157.87  137827673\n",
              "2021-11-19  157.650  161.020  156.5328  160.55  117305597\n",
              "\n",
              "[2993 rows x 5 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 191
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LpW8PqeNcPMh",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1758bed1-e5ff-436e-8905-55522aefd56a"
      },
      "source": [
        "# get the number of rows and columns\n",
        "df.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2993, 5)"
            ]
          },
          "metadata": {},
          "execution_count": 192
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "9ZlUr0M2cU_K",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 558
        },
        "outputId": "b8ecd2c8-974d-40f7-a48d-0eb9f4e3de32"
      },
      "source": [
        "#visualize the closing price history\n",
        "plt.figure(figsize=(16,8))\n",
        "plt.title('Close Price history')\n",
        "plt.plot(df['close'])\n",
        "plt.xlabel('Date', fontsize=18)\n",
        "plt.ylabel('Close Price in USD', fontsize=18)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABCwAAAIdCAYAAAD25OyiAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXhU9dn/8c8s2UhCAiGENSAQwA0RAZdWpQIibohYUdBHQX5YxVbBulD7VKtU3HdcKG4oSH1QcV9qi6CCULWKCiL7vgYSsmdmcn5/xERm5szkTDKTWfJ+XZfXRWbOnPnO4pzzvc99319bUVGRIQAAAAAAgBhij/YAAAAAAAAAfBGwAAAAAAAAMYeABQAAAAAAiDkELAAAAAAAQMwhYAEAAAAAAGIOAQsAAAAAABBzCFgAABBFW7ZsUXZ2tq655ppoDyUmXHPNNcrOztaWLVuiPRR9+umnIX828+bNU3Z2tubNmxfBkQEA0DIQsAAAIMzWrVunW265Raeccory8/OVm5ur3r1766KLLtJzzz2nsrKyaA8xbM455xxlZ2d7/de5c2edcsopuuuuu1RUVBTtIcaVmTNnEvAAAOBnzmgPAACARHLffffpnnvuUU1NjQYOHKhLLrlEmZmZ2rt3r5YtW6Zp06bp8ccf13//+99oDzWsLr30UuXn58swDO3Zs0fvv/++HnzwQS1atEj/+te/lJ2dbWk/t99+u6ZOnapOnTpFeMSRce6552rQoEHKy8uL9lAAAIh7BCwAAAiTBx98UHfffbc6d+6s559/XoMHD/bbZvHixZoxY0YURhdZ48aN06mnnlr/94wZMzRs2DCtXbtWs2fP1s0332xpPx06dFCHDh0iNcyIy8rKUlZWVrSHAQBAQqAkBACAMNiyZYvuueceJSUl6R//+IdpsEKSfvOb3+i9996ztM+9e/fq5ptv1nHHHaf27dvriCOO0NixY/X555/7bWsYhl555RWNGDFCvXr1Ul5eno466iidd955evHFF/22Ly4u1t/+9jedfPLJ6tixo7p06aKzzjpLixYtCu2FB5CZmalx48ZJkr766qv627Ozs3XsscequLhYt956q4455hjl5OToySeflBS8h8XXX3+tiRMn6sgjj6wvsznvvPM0f/58v22/+eYbTZw4UX379lVubq769OmjyZMna+PGjY16PVu2bNHEiRPVo0cP5eXlaciQIfrggw/8tgvUw+L777/XpEmT1K9fP+Xl5alHjx465ZRTdOONN6q4uFhSbXnNvffeK0maMmWKV5nN4e9HSUmJZsyYUZ/JkZ+fr/POO0/vvvuu6bizs7N1zjnnaNeuXZoyZYr69Omjtm3b6p133tHw4cPVpk0bbd682fR1P/fcc8rOzk7IIBsAIPaRYQEAQBjMmzdPLpdLF154oY455pig26akpDS4v61bt2rkyJHasWOHfvWrX+nCCy/U7t27tWjRIv3zn//U448/rvHjx9dvf9ddd+mhhx5Sfn6+Ro0apaysLO3Zs0fff/+9FixYoCuuuKJ+2507d+q8887Thg0bdPLJJ+vKK69UeXm5PvroI1155ZW65ZZbNH369Ma/GT8zDMP09urqap1//vkqLi7W8OHDlZaWps6dOwfd19y5czV16lTZ7XadddZZKigoUGFhob799ls99dRT9cERSXr11Vd17bXXKjk5WSNHjlTnzp21ceNGvfbaa/rggw/0zjvvqF+/fpZfx7Zt2zR06FB1795dY8eO1cGDB/XGG29o3LhxWrRokU477bSgj//+++81bNgw2Ww2jRgxQkcccYRKS0u1detWzZ8/X1OmTFFWVlb9a/j888919tln69hjj63fR13WRnFxsUaOHKnVq1erX79++t3vfqfi4mItWrRI48eP1/Tp03XLLbf4jeHgwYMaPny4WrdurVGjRskwDLVp00YTJ07Uf/7zH82dO1d/+ctf/B73/PPPy263e31/AABoLgQsAAAIgy+++EKSNGTIkLDsb9q0adqxY4duvfVW3XrrrfW3X3fddRo2bJimTZumIUOG1E/0n3/+eXXs2FHLly9Xenq6174KCwu9/r7mmmu0ceNGzZkzRxdddFH97YcOHdK5556r++67T+eee67XhDlUJSUl9ZkPAwcO9Lpvz549OvLII/X++++rVatWDe7rxx9/1LRp05Senq73339fRx99tNf927dvr//3xo0b9fvf/15dunTRe++959UL49NPP9UFF1yg3//+91qyZInl1/LZZ5/5fQ6//e1vNWbMGD3++OMNBixeeeUVVVZW6uWXX9a5557rdV9JSYmSk5MlSePHj9fWrVv1+eef65xzzvEKSNX561//qtWrV2v8+PF64oknZLPZJEk33XSTzjjjDN1zzz0aPny4BgwY4PW41atXa+zYsZo1a5aczl9O/0444QTddtttevnllzV9+nQlJSXV3/fll1/qu+++04gRI9S1a1eL7xYAAOFDSQgAAGGwZ88eSQpLs8idO3fq448/VufOnTVt2jSv+44++mhNnDhRVVVV+sc//uF1X1JSktdktE5OTk79v3/44QctWbJE55xzjlewQpJat26tW2+9VYZh6P/+7/9CGvP8+fM1c+ZM3X333br++us1cOBA/fTTT+rRo4f+3//7f37b33XXXZaCFZL07LPPyu12649//KNfsEKSunTp4rVtVVWV7r77br/P4tRTT9XIkSP17bff6scff7T82rp27aqbbrrJ67ahQ4eqS5cuXuUuDUlLS/O7LTMz01LGjSS5XC4tWLBArVq10p133lkfrJBU/10xDENz5871e2xycrJmzJjh9/1ITU3VZZddpr179/qVlDz//POSpAkTJlgaHwAA4UaGBQAAMebbb7+VJJ144on1V98PN2TIEM2aNat+O6n2iv/s2bM1ePBgXXDBBTr55JN14oknqk2bNl6PXbFihaTaK/szZ87023ddNsbatWtDGvMrr7xS/+9WrVqpe/fuGj9+vP7whz/4rRCSmpraYNnM4b788ktJ0rBhwxrctu71LVu2zOv9qbNv3z5Jta+vb9++lp7/2GOPlcPh8Lu9S5cuWrlyZYOPv/DCC/X0009r/PjxOv/883Xaaadp8ODB6t27t6Xnr/PTTz+pvLxcAwcO9ApC1anL7jF73XXL65qZOHGinnjiCT3//PO64IILJNWWnrzxxhvq0qWLzjzzzJDGCQBAuBCwAAAgDPLy8rR27Vrt3Lmzyfs6dOiQJKl9+/YBn0tSfbNGSZo5c6Z69Oih+fPn67HHHtOjjz4qu92u008/XXfeeWd9eceBAwckSUuWLAlaFlFWVhbSmN9++22vVUKCadeunVd2QEPqXqeV7JW61/fEE08E3S6U1xdo1Q+Hw6GampoGH3/CCSfogw8+0IMPPqh33nlHr776qqTaIMINN9ygiRMnWhpHY74XdQI9RpK6d++uoUOH6uOPP9aGDRvUs2dPLViwQOXl5fV9QwAAiAaOQAAAhMFJJ50kSSH1RgikdevWkmpXCTFTV35St51UO3n+3e9+p6VLl2rDhg2aP3++Lr74Yn3yyScaPXp0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ys3NVX5+vubMmaONGze27osFAABxocICAAAkjYsvvlhFRUV65513NHnyZH3xxRd6/fXX9ZOf/ET9+/dXIBDQP/7xD91xxx3avn27HnroIUnhkKKsrEzLli3TwoULNXjwYElS//79JUmBQECzZs3SunXrNGfOHF155ZU6cOCAnn76aZ199tl68803NWbMmDZ73QAAIBqBBQAASBojRoyQJG3btk2SNGXKFG3cuFGGYdRuc80112jevHl65plndOONNyovL08jR47UhAkTtGzZMk2bNk2nnHJKxH4ff/xxrVmzRi+//LJOO+202scvv/xynXzyybrlllu0cuXKVniFAAAgXkwJAQAASaNz586SpKqqKklSWlpabVjh9/tVWVmpiooKnXbaaTJNUxs2bIhrvy+++KIGDx6s0aNHq6KiovZfIBDQtGnTtHbtWnk8nmPzogAAQLNQYQEAAJLGgQMHJEmZmZmSpGAwqAcffFAvvPCCSkpKopY9dbvdce13y5Yt8ng8GjhwYIPbVFRUqE+fPs08cgAA0NIILAAAQNL44osvJEmFhYWSpJtuukmPP/64LrjgAi1cuFDZ2dlyOp3auHGjbr/9dpmmGdd+LcvS8OHDtWjRoga36dGjx9G/AAAA0GIILAAAQNJ49tlnJUlnnnmmJGn58uU6+eST9ec//zliu5KSkqjn1u1zUd+AAQNUUVGhU089VTYbM2IBAGgPOGMDAIA2FwqFdMstt6ioqEhnnnmmJk2aJCm83Gn9aSAHDx7Uo48+GrWPTp06SZIqKyujfjZ37lyVlZVpyZIlMX9/eXn50b4EAADQwqiwAAAArWrjxo1avny5JKm6ulpbt27VypUrtWPHDk2fPl1PPPFE7bbnnnuunnrqKV122WWaNm2aysvL9eyzz6pbt25R+x07dqxsNpseeOABud1uderUSQUFBRo/frzmz5+v9957T7feeqs++OADnXrqqcrMzNTOnTv1/vvvy+Vy6Y033mi19wAAADTNcLvdVtObAQAAHJ3Vq1dr5syZtf+32WzKyMhQr169NHr0aM2ePVunn356xHNqamq0ePFivfLKK9qzZ4969+6tSy65RGPHjtW5556rJUuW6KKLLqrd/vnnn9fDDz+skpISBQIBzZ07V0uXLpUUbuD55JNPavny5dq8ebMkKS8vT+PGjdPcuXM1ffr0VngXAABAvAgsAAAAAABA0qGHBQAAAAAASDoEFgAAAAAAIOkQWAAAAAAAgKRDYAEAAAAAAJIOgQUAAAAAAEg6BBYAAAAAACDpEFgAAAAAAICkQ2ABAAAAAACSDoEFAAAAAABIOgQWAAAAAAAg6fwv1lmWnEXnVKsAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1152x576 with 1 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "bDwuJ3ELdlpQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d2da30c4-869b-4638-a570-cc55c0df177f"
      },
      "source": [
        "# Create a new dataframe with only the close column\n",
        "data = df.filter(['close'])\n",
        "#convert the dataframe to numpy array\n",
        "dataset = data.values\n",
        "#get the number of rows to train the model on\n",
        "training_data_len = math.ceil(len(dataset) * .8)\n",
        "\n",
        "training_data_len"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "2395"
            ]
          },
          "metadata": {},
          "execution_count": 194
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jeDxEhGqeIOW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "062457cd-2949-4417-f08d-7f5af10a3fbd"
      },
      "source": [
        "# Scale the data\n",
        "scaler = MinMaxScaler(feature_range=(0,1))\n",
        "scaled_data = scaler.fit_transform(dataset)\n",
        "\n",
        "scaled_data"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[0.20223268],\n",
              "       [0.20283744],\n",
              "       [0.1972639 ],\n",
              "       ...,\n",
              "       [0.1033147 ],\n",
              "       [0.11047367],\n",
              "       [0.11485404]])"
            ]
          },
          "metadata": {},
          "execution_count": 195
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1gWSRuVu0NOI",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bba4f3de-8471-48d1-b95c-0caa546cf6d6"
      },
      "source": [
        "#Create the training dataset\n",
        "#create the scaled training data set\n",
        "train_data = scaled_data[0:training_data_len , :]\n",
        "#split the data into x_train and y_train data sets\n",
        "x_train = []\n",
        "y_train = []\n",
        "\n",
        "for i in range(60, len(train_data)):\n",
        "  x_train.append(train_data[i-60:i, 0])\n",
        "  y_train.append(train_data[i, 0])\n",
        "  if i<= 61:\n",
        "    print(x_train)\n",
        "    print(y_train)\n",
        "    print()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[array([0.20223268, 0.20283744, 0.1972639 , 0.19662646, 0.19891471,\n",
            "       0.19585826, 0.19195188, 0.19674087, 0.19474682, 0.18902618,\n",
            "       0.20391618, 0.19849792, 0.19252721, 0.17565624, 0.18435978,\n",
            "       0.18904253, 0.19221993, 0.17817332, 0.16636102, 0.17072015,\n",
            "       0.17256709, 0.17807525, 0.16633977, 0.17191331, 0.16972312,\n",
            "       0.17310647, 0.17135105, 0.17715995, 0.17995489, 0.18489098,\n",
            "       0.18350168, 0.18411951, 0.18206335, 0.18001373, 0.17452682,\n",
            "       0.180406  , 0.18260273, 0.18688503, 0.19402766, 0.19379883,\n",
            "       0.19458337, 0.19683894, 0.21030695, 0.21051943, 0.21695924,\n",
            "       0.21993397, 0.22101272, 0.22281063, 0.2182995 , 0.21929653,\n",
            "       0.21875715, 0.21962342, 0.21570053, 0.21978687, 0.22568729,\n",
            "       0.22733811, 0.22289235, 0.22983884, 0.2322742 , 0.23792128])]\n",
            "[0.23654015887025592]\n",
            "\n",
            "[array([0.20223268, 0.20283744, 0.1972639 , 0.19662646, 0.19891471,\n",
            "       0.19585826, 0.19195188, 0.19674087, 0.19474682, 0.18902618,\n",
            "       0.20391618, 0.19849792, 0.19252721, 0.17565624, 0.18435978,\n",
            "       0.18904253, 0.19221993, 0.17817332, 0.16636102, 0.17072015,\n",
            "       0.17256709, 0.17807525, 0.16633977, 0.17191331, 0.16972312,\n",
            "       0.17310647, 0.17135105, 0.17715995, 0.17995489, 0.18489098,\n",
            "       0.18350168, 0.18411951, 0.18206335, 0.18001373, 0.17452682,\n",
            "       0.180406  , 0.18260273, 0.18688503, 0.19402766, 0.19379883,\n",
            "       0.19458337, 0.19683894, 0.21030695, 0.21051943, 0.21695924,\n",
            "       0.21993397, 0.22101272, 0.22281063, 0.2182995 , 0.21929653,\n",
            "       0.21875715, 0.21962342, 0.21570053, 0.21978687, 0.22568729,\n",
            "       0.22733811, 0.22289235, 0.22983884, 0.2322742 , 0.23792128]), array([0.20283744, 0.1972639 , 0.19662646, 0.19891471, 0.19585826,\n",
            "       0.19195188, 0.19674087, 0.19474682, 0.18902618, 0.20391618,\n",
            "       0.19849792, 0.19252721, 0.17565624, 0.18435978, 0.18904253,\n",
            "       0.19221993, 0.17817332, 0.16636102, 0.17072015, 0.17256709,\n",
            "       0.17807525, 0.16633977, 0.17191331, 0.16972312, 0.17310647,\n",
            "       0.17135105, 0.17715995, 0.17995489, 0.18489098, 0.18350168,\n",
            "       0.18411951, 0.18206335, 0.18001373, 0.17452682, 0.180406  ,\n",
            "       0.18260273, 0.18688503, 0.19402766, 0.19379883, 0.19458337,\n",
            "       0.19683894, 0.21030695, 0.21051943, 0.21695924, 0.21993397,\n",
            "       0.22101272, 0.22281063, 0.2182995 , 0.21929653, 0.21875715,\n",
            "       0.21962342, 0.21570053, 0.21978687, 0.22568729, 0.22733811,\n",
            "       0.22289235, 0.22983884, 0.2322742 , 0.23792128, 0.23654016])]\n",
            "[0.23654015887025592, 0.23812559249452453]\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "6PTatXXR1mA2"
      },
      "source": [
        "#Convert the x_train and y_train data set to numpy arrays\n",
        "x_train, y_train = np.array(x_train),np.array(y_train)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "hDQYYQE41ywE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8f911f5c-17f5-4f06-f06f-e92f16617ba5"
      },
      "source": [
        "#reshape the data\n",
        "print(x_train)\n",
        "print(\"x_train.shape[0]\",x_train.shape[0])\n",
        "print(\"x_train.shape[1]\",x_train.shape[1])\n",
        "x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))\n",
        "x_train.shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[0.20223268 0.20283744 0.1972639  ... 0.22983884 0.2322742  0.23792128]\n",
            " [0.20283744 0.1972639  0.19662646 ... 0.2322742  0.23792128 0.23654016]\n",
            " [0.1972639  0.19662646 0.19891471 ... 0.23792128 0.23654016 0.23812559]\n",
            " ...\n",
            " [0.18034716 0.1776176  0.17748684 ... 0.18379589 0.18654179 0.18624759]\n",
            " [0.1776176  0.17748684 0.17807525 ... 0.18654179 0.18624759 0.17936648]\n",
            " [0.17748684 0.17807525 0.17810794 ... 0.18624759 0.17936648 0.18136053]]\n",
            "x_train.shape[0] 2335\n",
            "x_train.shape[1] 60\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(2335, 60, 1)"
            ]
          },
          "metadata": {},
          "execution_count": 198
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7RpX--bk3TV4"
      },
      "source": [
        "#build the LSTM model\n",
        "model = Sequential()\n",
        "model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1],1)))\n",
        "model.add(LSTM(50, return_sequences=False))\n",
        "model.add(Dense(25))\n",
        "model.add(Dense(1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gEemdqes31bE"
      },
      "source": [
        "#compile the model\n",
        "model.compile(optimizer='adam', loss='mean_squared_error')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "35ZG6o9j4AJo",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "89e8195b-c385-4a76-a2df-054b9ed54d5f"
      },
      "source": [
        "#train the model\n",
        "\n",
        "model.fit(x_train, y_train, batch_size=1, epochs=1)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2335/2335 [==============================] - 48s 20ms/step - loss: 0.0010\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<keras.callbacks.History at 0x7f73c87a5d50>"
            ]
          },
          "metadata": {},
          "execution_count": 258
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "-L9dGS004nlF"
      },
      "source": [
        "#Create the testing dataset\n",
        "#create a new array containing scaled values from index 1543 to 2003\n",
        "test_data = scaled_data[training_data_len - 60: , :]\n",
        "#create the data sets x_test and y_test\n",
        "x_test = []\n",
        "y_test = dataset[training_data_len:, :]\n",
        "\n",
        "for i in range(60, len(test_data)):\n",
        "  x_test.append(test_data[i-60:i,0])"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "eCwWYmc25Z1T"
      },
      "source": [
        "#convert the data to a numpy array\n",
        "x_test = np.array(x_test)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cMqNt0wW5hO_"
      },
      "source": [
        "#Reshape the data\n",
        "x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ykTvGpqQ55Ce"
      },
      "source": [
        "#get the models predicted price values\n",
        "predictions = model.predict(x_test)\n",
        "predictions = scaler.inverse_transform(predictions)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oV0W0jDk6U0b",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6897b35b-5645-4391-b2a8-b02d40a87785"
      },
      "source": [
        "#getting the root mean squared error (RMSE)\n",
        "rmse=np.sqrt(np.mean(((predictions- y_test)**2)))\n",
        "rmse"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "15.94853394962508"
            ]
          },
          "metadata": {},
          "execution_count": 263
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JLG_KB1E8WYl",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 660
        },
        "outputId": "bcf60a12-eae2-4c9c-ed52-275b5f9f7bdb"
      },
      "source": [
        "#Plot data\n",
        "train = data[:training_data_len]\n",
        "valid = data[training_data_len:]\n",
        "valid['Predictions'] = predictions\n",
        "#Visualize the data\n",
        "plt.figure(figsize=(16,8))\n",
        "plt.title('Model')\n",
        "plt.xlabel('date', fontsize=18)\n",
        "plt.ylabel('Closed Price USD',fontsize=18)\n",
        "plt.plot(train['close'])\n",
        "plt.plot(valid[['close','Predictions']])\n",
        "plt.legend(['Train','Val','Predictions'],loc='lower right')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame.\n",
            "Try using .loc[row_indexer,col_indexer] = value instead\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  after removing the cwd from sys.path.\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAABCwAAAIdCAYAAAD25OyiAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdeXxU5d338e+ZLTsEQggCRgUiiIIICNZapS5FRKu4VNve1XI/FtzaSq1WtNXWuz5UtCqgot5qqwhuVG0Vq9WKVq2C1UdRQQTZtyQEAtkzy3n+QCIzc87kTDKTWfJ5v1591cyczFyZGWbm+p7f9buM2tpaUwAAAAAAAGnEleoBAAAAAAAARCKwAAAAAAAAaYfAAgAAAAAApB0CCwAAAAAAkHYILAAAAAAAQNohsAAAAAAAAGmHwAIAAGSdjRs3qri4WJdffnla3A4AAIgfgQUAAOi04uJiFRcXq1evXlq/fr3tceecc07bsY888kgXjhAAAGQaAgsAAJAQHo9Hpmnqscces7x+w4YNevPNN+XxeLp4ZAAAIBMRWAAAgITo3bu3jj32WC1atEiBQCDq+gULFsg0TZ1++ukpGB0AAMg0BBYAACBhLr74YlVWVurvf/972OWBQEALFy7UmDFjdOSRR9r+/oYNG3TFFVdo+PDhKi0tVUVFhX784x/r008/tTy+rq5ON9xwg4YPH66ysjIde+yxuueee2Sapu19NDc3a968eTrppJM0YMAA9e/fXxMmTNAjjzwS8/cAAEDXIrAAAAAJc+6556qoqChqWcgrr7yiHTt26JJLLrH93Y8++kgnnXSSnnjiCY0YMUI//elPdcIJJ+jFF1/Uqaeeqtdffz3s+JaWFp199tm67777VFxcrMsuu0wnnHCC7rjjDs2cOdPyPurq6jR58mT95je/kWma+sEPfqAf/vCH2rt3r37xi1/oiiuu6PyDAAAAEoJFpAAAIGEKCgp0/vnn69FHH9XmzZt18MEHS5Iee+wxFRYW6txzz9W8efOifs80TV122WXas2eP7rvvPv3gBz9ou+6NN97QlClTNG3aNK1YsUL5+fmSpHvuuUcffvihzjjjDD3++ONyufadh5kxY4YmTJhgOb4bbrhBH3zwgX7729/q6quvbru8paVFP/rRj/TEE0/ou9/9riZNmpSohwQAAHQQFRYAACChLrnkEoVCIT3++OOSpK1bt+q1117Teeedp8LCQsvfWbZsmT7//HONHj06LKyQpAkTJujMM8/Uzp079dJLL7VdvnDhQhmGod/97ndtYYUklZeXa/r06VH3sXv3bj3xxBMaOXJkWFghSTk5ObrpppskSU899VTH/nAAAJBQVFgAAICEGjVqlEaOHKmFCxfquuuu04IFCxQMBmMuB/n4448lSSeeeKLl9RMmTNALL7ygjz/+WOeff77q6uq0bt069evXTxUVFVHHf/Ob34y67IMPPlAgEJDL5dKsWbOirt/fKPSLL75w9HcCAIDkIrAAAAAJd8kll+iaa67RK6+8oscff1xHHXWURo8ebXv83r17JUl9+/a1vL6srEyStGfPnrDjS0tLLY+3up1du3ZJ2tcr46OPPrIdS319ve11AACg67AkBAAAJNwFF1yg/Px8XXvttdqyZYt+/OMfxzy+R48ekqSqqirL6ysrK8OO2///1dXVlsdb3c7+35k2bZpqa2tt/7dixYr2/0AAAJB0BBYAACDhevTooSlTpmjr1q3Kz8/XBRdcEPP4o48+WpL01ltvWV7/5ptvStq33ESSioqKNGjQIFVWVmrt2rVRx7/zzjtRl40dO1Yul0vvvvtuXH8LAABIDQILAACQFDfccIMef/xxLV68WD179ox57Pjx4zV06FB98MEHUU0v33zzTb3wwgsqKSnRGWec0Xb5D3/4Q5mmqZtuukmhUKjt8k2bNumBBx6Iuo8+ffrowgsv1CeffKJZs2a19aw40NatW+lhAQBAmqCHBQAASIoBAwZowIABjo41DEPz58/XOeeco8suu0zPPfecjjzySK1fv15/+9vf5PP5dP/997dtaSpJV111lZYsWaKXXnpJ3/rWt3Tqqadq7969eu655/SNb3xDf//736PuZ/bs2Vq3bp1uu+02PfXUUzr++ONVVlbWVqnx/vvv69Zbb9Xhhx+esMcBAAB0DIEFAABIC6NHj9Ybb7yh22+/XW+88Yb++c9/qmfPnpo8ebKuueYajRw5Muz4nJwcPf/88/rDH/6g5557Tvfff7/Ky8t1zTXX6KyzzrIMLIqKivTiiy9qwYIFeuaZZ/Tiiy+qublZpaWlOuSQQ3TzzTdrypQpXfUnAwCAGIza2loz1YMAAAAAAAA4ED0sAAAAAABA2iGwAAAA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AbmmjAQkAL0sdpk5OAG9MQAAQGrtDyskyd/Ppfyaj1WX/80UjgiZLmZg8dlnn+mee+7RSSedpD//+c8qLi6OOqa2tlYXX3yx5s6dqwsuuEDDhw93dMdjx47Vfffdp4qKCu3cuVO33367vvOd7+i9995TZWWlJKm0tDTsd0pLS7V9+3ZJUlVVlUpKStoqPSTJMAz16dNHVVVVMe97zZo1jsYIdHdWPSzKmVjKazNp75HGjSPf2R1/OeaWyOUgCXrux/QMT8J6ekyd2od0DACATGZafD/K271adQcTWHS1TJrvtldoEPPr9cKFC1VYWGgbVkhScXGxHn30UR199NFatGiRfv/73zsa2GmnnRb289ixYzVq1CgtWrRIxx57rKPb6KhMq77IpBccsotVD4tjehJY2PWwSGef18c/6Oj+FYmpIBmUb2rqQL/+tMUrt2HqhiGt7BICAEAGM4IBtQyM/q7hCvpTMBpk2nw3lpjfYN9//32dddZZtmHFfr169dKZZ56p9957r8MDKSws1LBhw7Ru3bq2vhbV1dVhx1RXV6tv376SpL59+6qmpkam+fUXaNM0tXPnzrZjAHSOVQ+LS8v54LHrYZHOUU7Q6rSHhQOXjmxuSvwOIftdcahffx/XqH+Ob9LJXVhdYabvqh0AADKWEfCrYZTF2Qd3S/RlQBxiBhbr16/XiBEjHN3QyJEjtWHDhg4PpLm5WWvWrFFZWZkOOeQQlZWVaenSpWHXv/vuu209K8aNG6f6+notX7687Zjly5eroaEhrK8FgI57syZ8O5CxPYPql8OMz66HRUf6RHSVRoeZQOsBqUuyloTs18cnFSS5siIRPTcAAEBsRjCoYH70h66RvyMFo0E2iflVce/eve1WV+xXXFysuro6x3f861//WqeffroGDhzY1sOisbFR3//+92UYhi6//HLdeeedqqio0JAhQ3THHXeooKBA559/viRp6NChOvXUUzVjxgzdfffdkqQZM2Zo4sSJWVUCA6RSdWv4B89xvegzINnvkpLGeYUabXZ8ifS3So8GF4Q0ukdIWyOabiZqSQgAAMguRigow7T6LpSB62iRVmIGFn6/X263O9YhbVwul/x+56Xi27Zt06WXXqqamhr16dNHY8eO1auvvqry8nJJ0s9//nM1NTXp2muvVW1trcaMGaNnn31WRUVFbbfx0EMP6brr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HAcWV111lc455xxNnDhRv/rVrzRixAhJ0ieffKLbbrtN27Zt07x585I2UKA7qw9Iv1/j05oGl77XP6AL+wcs+0sE4uw5ERlYnNArqKsPY3KbSLluaXJZMOYx/XJC2tESnU7kue0TKK9LKvaYqrXZHeQH/f26qH9APb1m2HaoAAAAnRW9JISzXEgOx4HFiSeeqD/+8Y+6/vrrNXXq1LDrvF6vbr/9dk2YMCHR4wMg6X83efXPmn3/XO9Y59NRRSG1WgUWZnzpdmRgUeKjYD4VPDZPW0E7QUP/3JBq660POrMsEPcSIQAAACeim7YTWCA54tpSYOrUqZo4caKee+45rV+/XpI0ePBgnX322erfv39SBghAWrTNG/bzM9s9Gl4YnVgE45yfrmsMnykflMO2IKlgF1jkx6iwkKRSm4BpbM+gKhxsm9ot8bAAANBpZmQPi/0VFpG7OXBLWAAAIABJREFUhPDBi06Kew/E/v3768orr0zGWABYCFm8zzcErbccjbeHxbrG8DT8cIsQBMnntg0sYv9ers31lx3Csh4AAJA80RUWX32ZIZ9AglG7A6Q5q2BCpmyWhMR32ztaIiss+JRJhY5WWNjtIpIb5/a22YwWYAAAJEFE002TaSWSxLbC4sorr5RhGJozZ47cbrejqgrDMHTPPfckdIBAd9dsEUw0hQw1W+wS8o9qj75dEtSd63xyG6auG+zX8CLrqomXq9zaHtHosUfcNVdIBI9hHTC018PCLphgdxcAAJBckd8v+fKB5LCdnixatEiGYejOO++U2+3WokWL2r0xAgsg8ZqC0cHEl42GBuZGH1vjNzTtk6+v+MOX0mOjWqKOM03prvW+qMt7eDgznwoem894ux4V++XZBBp2S0UAAAASIvJky1c9LPgmiUSzjcJ2796tXbt2yefztf3c3v927drV4YHceeedKi4u1rXXXtt2mWmamjVrloYNG6Z+/fpp8uTJWrVqVdjv1dbWatq0aSovL1d5ebmmTZum2traDo8D6AqNQenGz32atDxXd3zpjbmUw6rCYmerS9sstsGMtKreLdPitvcGpF3+6CCEiW5q2C0JsQsy9rOrsGBJiD0eGQAAOs8wbZpushgTCeaodicYDGrz5s3avXt3Ugbx/vvv689//rOOPPLIsMvnzJmje++9V7fddptef/11lZaWasqUKaqrq2s7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AXxhcBpoBQ+ps5GtQoxoWabvjojIcSY/P5/Zs2djt9vJzs7G4agcipmUFDn6LhwOk5GRwWuvvRZ1nJSUlGPe1yN1aKHQQ1/fwccO1rlITU3lo48+YunSpXzwwQc88MAD3H333SxatIg2bdrUa58UWIiIyHHHpfGFchQUVoiI1K/6qClxrCUmJtKxY8date3Tpw+7d+/GNE3at28fs02XLl1YuXJlxLbPPvus2uP26tWLxYsXM2nSpJiPO51OQqHql8zu1q1b1FKpy5Yto127dhGLW9TEbrczcuRIRo4cycyZM+ncuTPvvPMOU6ZMqfUxavU89Xo0ERERERERkePYiBEjGDhwIJdeeim33347nTt3Zvfu3SxatIgRI0aQn5/P5ZdfztixY3n44YcZO3Ys//3vf3nrrbeqPe4NN9zAJZdcQseOHTnnnHOwLIsPP/yQyZMnk5SURE5ODp988gnbtm3D5XLRqlWrqGNce+21nHrqqcyaNYvzzjuPlStXMnfuXG6//fZav763336bjRs3MmTIENLT01m8eDHFxcV07dq1zu9VTXSPSURERERERKSeGIbBs88+y7Bhw5g+fTrDhg1j6tSpfP/992RnZwPQr18/7r//fubNm8epp57Km2++ybRp06o97umnn85TTz3FokWLOOOMMxg/fjxLly7FNMsv63/zm9+wbds2+vbtS6dOnWIe4+STT+bpp59m4cKFDB48mN///vfceOONFQU2ayMtLY0333yTcePGMXDgQObMmcPDDz/MkCFDan2M2jI8Ho/W2Ypz69evb+wuiIiIiIiIHJWEhISI4pRybCQkJDR2F6IUFBSQlpZW5/00wkJERERERERE4o4CCxERERERERGJOwosRERERERERCTuKLAQERERERERkbijwEJERERERERE4o4CCxERERERERGJOwosRERERERE5JgLBAIUFxdjWVZjd0UakN/vxzSPLHqw13NfRERERERERKKEQiE8Hg8lJSVHfAErNWvRokVjdyGCaZqkpKQc0b4KLERERERERKRBWJaF3+9v7G40azk5OY3dhXqjWEtERERERERE4o4CCxERERERERGJOwosRERERERERCTuKLAQERERERERkbhjeDwerSkjIiIiIiIiInFFIyxEREREREREJO4osBARERERERGRuKPAQkRERERERETijgILEREREREREYk7CixEREREREREJO7YG7sD8aJjx47s37+/sbshIiIiIiIi0qxcd911/OEPf6jzfhphcYDH42nsLoiIiIiIiIg0G4ZhAPD6668TDofrvL8CiwP279+Px+Op+LdkyZLG7pKIiIiIiIhIk2VZFna7nc2bN/PRRx/VeX8FFlX49ttvG7sLIiIiIiIiIk1aKBTCNE2WLVtW530VWFRh6tSpjd0FERERERERkSbNsixCoRA7d+6s874KLGLIzs4+ovk1IiIiIiIiIhIpLS0N06x7/KDA4jDZ2dl4vd7G7oaIiIiIiIhIk+dyuSgqKqJDhw513leBxSEUVoiIiIiIiIjUn3A4TDgcZvTo0XXe134M+tMkZWVl4fP5GrsbIiIiIiIiIs1GIBBg2LBhdOnSpc77Gh6PxzoGfWpy3G53Y3dBREREREREpFlp164da9aswTCMOu+rERYHeDyexu6CiIiIiIiIiBygGhYiIiIiIiIiEncUWIiIiIiIiIhI3FFgISIiIiIiIiJxR4GFiIiIiIiIiMQdBRYiIiIiIiIiEncUWIiIiIiIiIhI3FFgISIiIo2ud+/ejBkzprG7ISIiInFEgYWIiIg0eY888gjPPvtsY3dDRERE6pECCxEREWnyHn30UebPn9/Y3RAREZF6pMBCREREREREROKOAgsRERFpMFu2bGHKlCnk5uaSk5PDBRdcwMaNG2O2ffnll7nwwgvp1asXmZmZ5OXlcfHFF7NmzZqIdm63m82bN7N06VLcbnfFv02bNlW0WbVqFRMnTiQvL4/MzEz69+/P7NmzCQaDx/T1ioiIyJEzPB6P1didEBERkebP4/EwYsQItm7dymWXXUa3bt1YunQpK1asoKysjO7du/Pmm29WtB89ejTp6en07duXrKwsNm7cyNNPP00gEODDDz+kU6dOACxYsIDf/va3tGrVimnTplXsP3bsWJKTk3nnnXeYPHkyeXl5nH/++aSnp7N8+XIWLFjAWWedxbx58xr8vRAREZGaKbAQERGRBnHXXXdx//33M2fOHCZNmlSx/dZbb+Wxxx5j6NChEYFFSUkJycnJEcdYu3Ytw4cPZ/Lkyfz5z3+u2N67d29yc3Mj9gfwer306dOHTp06sXDhQux2e8Vjc+fO5bbbbmPhwoUMHz68vl+uiIiIHCVNCREREZEG8eabb5KZmclFF10Usf3GG2+M2f5gWGFZFoWFhezdu5fWrVvTuXNnPv3001o95/vvv8+uXbuYOHEiBQUF7N27t+LfmWeeWdFGRERE4o+95iYiIiIiR++HH37glFNOwWazRWzPzs4mLS0tqv3nn3/OPffcw5IlSygpKYl4rH379rV6znXr1gFw3XXXVdlm165dtTqWiIiINCwFFiIiIhJ3Nm/ezJgxY0hNTWXGjBl07tyZ5ORkDMNg5syZFBcX1+o4llU+8/Xuu++md+/eMdtkZ2fXW79FRESk/iiwEBERkQbRoUMHvv/+e0KhUMQoix07dlBQUBDR9o033qC4uJj58+czYsSIiMf27duH0+mM2GYYRsznzMvLAyApKYmf/OQn9fAqREREpKGohoWIiIg0iJ///Ofs2rWL5557LmL7gw8+GNX2YKBxcITEQfPmzWPnzp1R7VNSUti/f3/U9tNOO42MjAweeOCBmI+XlZVRVFRUp9chIiIiDUOrhIiIiEiD8Hg8DBs2jO3bt3PZZZfRvXt3lixZEnNZ040bNzJ06FDS0tK44oorcLvdfPzxx7z77ru43W6CwSBffvllxbGvv/56/vGPfzBt2jS6deuGaZqMGjWK5ORk3nvvPSZOnEhycjKTJk0iLy+PgoIC1q1bx8KFC3nmmWe0SoiIiEgcUmAhIiIiDWbz5s3cdtttFStzDBkyhD/96U+cffbZUcuSLl26lLvvvps1a9Zgmib5+fnceeedzJgxgx9//DEisNi9ezc333wzixcvpqCgAMuy+PzzzyuKc3799dc8+OCDLF68mD179uB2u+nYsSOnn346V1xxBenp6Q37RoiIiEiNFFiIiIiIiIiISNxRDQsRERERERERiTsKLEREREREREQk7iiwEBEREREREZG4o8BCREREREREROKOAgsRERERERERiTsKLEREREREREQk7iiwEBEREREREZG4o8BCREREREREROKOAgsRERERERERiTsKLEREREREREQk7vw/sGm7Z0kOQMIAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 1152x576 with 1 Axes>"
            ]
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2O46Y-828Xc_",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 419
        },
        "outputId": "acb45a19-6efe-4290-d57e-8a43c23f7832"
      },
      "source": [
        "#show the valid prediction prices\n",
        "valid"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>close</th>\n",
              "      <th>Predictions</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>2019-07-11</th>\n",
              "      <td>201.75</td>\n",
              "      <td>203.414490</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2019-07-12</th>\n",
              "      <td>203.30</td>\n",
              "      <td>202.261230</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2019-07-15</th>\n",
              "      <td>205.21</td>\n",
              "      <td>203.510986</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2019-07-16</th>\n",
              "      <td>204.50</td>\n",
              "      <td>205.350525</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2019-07-17</th>\n",
              "      <td>203.35</td>\n",
              "      <td>204.900070</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>...</th>\n",
              "      <td>...</td>\n",
              "      <td>...</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-15</th>\n",
              "      <td>150.00</td>\n",
              "      <td>150.196701</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-16</th>\n",
              "      <td>151.00</td>\n",
              "      <td>150.440918</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-17</th>\n",
              "      <td>153.49</td>\n",
              "      <td>151.322723</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-18</th>\n",
              "      <td>157.87</td>\n",
              "      <td>153.616577</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2021-11-19</th>\n",
              "      <td>160.55</td>\n",
              "      <td>157.741592</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>598 rows × 2 columns</p>\n",
              "</div>"
            ],
            "text/plain": [
              "             close  Predictions\n",
              "2019-07-11  201.75   203.414490\n",
              "2019-07-12  203.30   202.261230\n",
              "2019-07-15  205.21   203.510986\n",
              "2019-07-16  204.50   205.350525\n",
              "2019-07-17  203.35   204.900070\n",
              "...            ...          ...\n",
              "2021-11-15  150.00   150.196701\n",
              "2021-11-16  151.00   150.440918\n",
              "2021-11-17  153.49   151.322723\n",
              "2021-11-18  157.87   153.616577\n",
              "2021-11-19  160.55   157.741592\n",
              "\n",
              "[598 rows x 2 columns]"
            ]
          },
          "metadata": {},
          "execution_count": 265
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "EEgNeyZT8nSd",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "dd0b49ee-634f-425d-ef24-0fd55fcf7d9b"
      },
      "source": [
        "#Get the quote\n",
        "apple_quote = web.DataReader(\"AAPL\", \"av-daily\", start=datetime(2021,8,15),end=datetime(2021, 11, 22),api_key='YJNQH53YAREKNLE4')\n",
        "#create a new data frame\n",
        "new_df = apple_quote.filter(['close'])\n",
        "#get the last 60 day closing price value and convert the df to array\n",
        "last_60_days = new_df[-60:].values\n",
        "#scale the data to be values between 0 and 1\n",
        "last_60_days_scaled = scaler.transform(last_60_days)\n",
        "#create an empty list\n",
        "X_test=[]\n",
        "#append the past 60 days data \n",
        "X_test.append(last_60_days_scaled)\n",
        "#convert the X_test data set to a numpy array\n",
        "X_test = np.array(X_test)\n",
        "#Reshape the data\n",
        "X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))\n",
        "#Get the predicted scaled price\n",
        "pred_price = model.predict(X_test)\n",
        "#undo the scaling\n",
        "pred_price = scaler.inverse_transform(pred_price)\n",
        "print(pred_price)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[[160.61624]]\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3dx-hSnB-PP6",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "beec57f6-0bda-4497-880f-7c69ba060f6e"
      },
      "source": [
        "#Get the quote\n",
        "apple_quote2 = web.DataReader(\"AAPL\", \"av-daily\", start=datetime(2021,11, 17),end=datetime(2021, 11, 23),api_key='YJNQH53YAREKNLE4')\n",
        "print(apple_quote2['close'])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2021-11-17    153.49\n",
            "2021-11-18    157.87\n",
            "2021-11-19    160.55\n",
            "Name: close, dtype: float64\n"
          ]
        }
      ]
    }
  ]
}