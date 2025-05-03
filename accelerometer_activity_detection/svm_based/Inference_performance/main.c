/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2024 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "stm32f411e_discovery_accelerometer.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "svm_acc_motion.h"
#include "svm_acc_motion_data.h"

#include "ai_datatypes_defines.h"
#include "ai_platform.h"
/* for inference performance */

#define SAMPLES_PER_SEGMENT 10
#define FEATURES_PER_SAMPLE 3
#define TOTAL_SAMPLES_PER_SEGMENT (SAMPLES_PER_SEGMENT * FEATURES_PER_SAMPLE)
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* for inference performance */
typedef struct {
	ai_float features[TOTAL_SAMPLES_PER_SEGMENT];
    uint8_t true_label;
} TestSegment;

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define NUM_SEGMENTS 10

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

I2S_HandleTypeDef hi2s2;
I2S_HandleTypeDef hi2s3;

SPI_HandleTypeDef hspi1;

/* USER CODE BEGIN PV */
int8_t strTemp[100];
char strPrediction[100];
ai_handle accelerometer_motion;
AI_ALIGNED(4) ai_float aiInData[AI_SVM_ACC_MOTION_IN_1_SIZE];
static AI_ALIGNED(4) ai_float aiOutData[AI_SVM_ACC_MOTION_OUT_1_SIZE];
ai_u8 activations[AI_SVM_ACC_MOTION_DATA_ACTIVATIONS_SIZE];
ai_buffer *ai_input;
ai_buffer *ai_output;
const char *activities[2] = { "Idle", "UpDown" };
AI_ALIGNED(4) ai_float model_output;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C1_Init(void);
static void MX_I2S2_Init(void);
static void MX_I2S3_Init(void);
static void MX_SPI1_Init(void);
/* USER CODE BEGIN PFP */
static void AI_Init(void);
static void AI_Run(float *pIn, float *pOut);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
TestSegment test_segments[] = {
    {
        {
            0.027322404371585, -0.087719298245614, -0.013732833957553,
            -0.054644808743169, -0.12280701754386, -0.013732833957553,
            0.021857923497268, -0.114035087719298, 0.008739076154807,
            -0.005464480874317, -0.105263157894737, -0.008739076154807,
            0.016393442622951, -0.12280701754386, 0.016229712858926,
            -0.049180327868853, -0.096491228070175, 0.0187265917603,
            0.021857923497268, -0.271929824561403, -0.0187265917603,
            0.027322404371585, -0.280701754385965, -0.016229712858926,
            -0.054644808743169, -0.263157894736842, 0.016229712858926,
            -0.054644808743169, -0.096491228070175, -0.016229712858926
        },
        0
    },
    {
        {
            -0.054644808743169, -0.105263157894737, 0.013732833957553,
            -0.049180327868853, -0.087719298245614, -0.01123595505618,
            0.021857923497268, -0.114035087719298, -0.01123595505618,
            0.027322404371585, -0.271929824561403, -0.0187265917603,
            0.016393442622951, -0.114035087719298, 0.0187265917603,
            0.027322404371585, -0.271929824561403, 0.006242197253433,
            0.021857923497268, -0.105263157894737, -0.0187265917603,
            0.021857923497268, -0.105263157894737, -0.016229712858926,
            0.010928961748634, -0.105263157894737, -0.013732833957553,
            0.016393442622951, -0.096491228070175, 0.01123595505618
        },
        0
    },
    {
        {
            0.021857923497268, -0.078947368421053, 0.001248439450687,
            -0.043715846994536, -0.280701754385965, -0.013732833957553,
            -0.054644808743169, -0.12280701754386, 0.013732833957553,
            -0.049180327868853, -0.114035087719298, -0.016229712858926,
            -0.049180327868853, -0.271929824561403, -0.016229712858926,
            0.021857923497268, -0.280701754385965, 0.006242197253433,
            -0.054644808743169, -0.096491228070175, -0.008739076154807,
            0.027322404371585, -0.271929824561403, -0.0187265917603,
            -0.043715846994536, -0.114035087719298, -0.013732833957553,
            0.005464480874317, -0.271929824561403, -0.013732833957553
        },
        0
    },
    {
        {
            0.021857923497268, -0.096491228070175, 0.016229712858926,
            0.021857923497268, -0.271929824561403, 0.0187265917603,
            0.021857923497268, -0.12280701754386, -0.016229712858926,
            -0.049180327868853, -0.096491228070175, 0.016229712858926,
            0.021857923497268, -0.271929824561403, -0.013732833957553,
            0.027322404371585, -0.271929824561403, 0.016229712858926,
            0.021857923497268, -0.114035087719298, 0.006242197253433,
            0.010928961748634, -0.096491228070175, -0.006242197253433,
            -0.049180327868853, -0.12280701754386, 0.0187265917603,
            0.021857923497268, -0.12280701754386, 0.016229712858926
        },
        0
    },
    {
        {
            0.076502732240437, -0.026315789473684, 0.340823970037453,
            0.273224043715847, -0.096491228070175, -0.041198501872659,
            -0.076502732240437, -0.18421052631579, -0.433208489388265,
            -0.491803278688525, -0.105263157894737, -0.265917602996255,
            -0.120218579234973, -0.535087719298246, -0.408239700374532,
            -0.398907103825137, -0.254385964912281, -0.418227215980025,
            -0.213114754098361, -0.307017543859649, -0.193508114856429,
            -0.562841530054645, 0.078947368421053, -0.17852684144819,
            -0.14207650273224, 0.245614035087719, 0.051186017478152,
            -0.284153005464481, 0.359649122807018, 0.250936329588015
        },
        1
    },
    {
        {
            0.049180327868853, -0.385964912280702, -0.255930087390761,
            -0.229508196721311, -0.105263157894737, -0.413233458177278,
            -0.065573770491803, -0.096491228070175, -0.677902621722846,
            -0.224043715846995, -0.307017543859649, -0.598002496878901,
            -0.131147540983607, -0.359649122807018, -0.146067415730337,
            0.038251366120219, -0.736842105263158, -0.103620474406991,
            0.038251366120219, -0.298245614035088, -0.01123595505618,
            0.256830601092896, 0.149122807017544, 0.208489388264669,
            0.442622950819672, 0.228070175438597, 0.493133583021224,
            0.792349726775956, 0.605263157894737, 0.225967540574282
        },
        1
    },
    {
        {
            0.349726775956284, -0.342105263157895, 0.493133583021224,
            0.027322404371585, -0.403508771929825, 0.533083645443196,
            -0.092896174863388, -0.359649122807018, 0.17852684144819,
            -0.169398907103825, -0.473684210526316, -0.118601747815231,
            -0.431693989071038, -0.263157894736842, -0.158551810237204,
            -0.672131147540984, -0.175438596491228, -0.35330836454432,
            -0.748633879781421, 0.219298245614035, -0.695380774032459,
            -0.584699453551913, 0, -0.602996254681648,
            -0.715846994535519, -0.026315789473684, -0.325842696629213,
            -0.245901639344262, -0.175438596491228, -0.203495630461923
        },
        1
    },
    {
        {
            -0.103825136612022, -0.333333333333333, 0.235955056179775,
            0.39344262295082, 0.298245614035088, 0.063670411985019,
            -0.480874316939891, -0.429824561403509, 0.021223470661673,
            -0.415300546448087, -0.342105263157895, -0.368289637952559,
            -0.491803278688525, 0.035087719298246, -0.598002496878901,
            -0.666666666666667, 0.184210526315789, -0.712858926342072,
            -1, 0.245614035087719, -0.640449438202247,
            -0.475409836065574, -0.052631578947369, -0.570536828963795,
            -0.371584699453552, 0.078947368421053, -0.168539325842697,
            -0.704918032786885, 0.447368421052632, 0.146067415730337
        },
        1
    },
    {
        {
            -0.032786885245902, -0.078947368421053, 0.880149812734082,
            0.153005464480874, 0.078947368421053, 0.765293383270911,
            0.07103825136612, -0.192982456140351, 0.410736579275905,
            -0.240437158469945, -0.12280701754386, 0.066167290886392,
            -0.191256830601093, 0.008771929824561, -0.108614232209738,
            -0.398907103825137, 0.087719298245614, -0.310861423220974,
            -0.475409836065574, 0.210526315789474, -0.348314606741573,
            -0.562841530054645, 0.5, -0.340823970037453,
            -0.245901639344262, 0.5, -0.523096129837703,
            -0.284153005464481, 0.307017543859649, -0.605493133583021
        },
        1
    },
    {
        {
            -0.131147540983607, 0.175438596491228, -0.006242197253433,
            -0.12568306010929, 0.105263157894737, -0.01123595505618,
            0.098360655737705, 0.456140350877193, 0.041198501872659,
            0.021857923497268, 0.12280701754386, -0.0187265917603,
            -0.027322404371585, -0.052631578947369, 0.013732833957553,
            0.049180327868853, 0.06140350877193, 0.036204744069913,
            -0.010928961748634, 0.035087719298246, 0.128589263420724,
            -0.016393442622951, -0.035087719298246, 0.0187265917603,
            -0.005464480874317, -0.017543859649123, 0.036204744069913,
            -0.038251366120219, -0.192982456140351, 0.001248439450687
        },
        1
    }
};

static void AI_Init(void) {
	ai_error err;

	const ai_handle act_addr[] = { activations };

	err = ai_svm_acc_motion_create_and_init(&accelerometer_motion, act_addr,
			NULL);
	if (err.type != AI_ERROR_NONE) {
		printf("ai_network_create error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
	ai_input = ai_svm_acc_motion_inputs_get(accelerometer_motion, NULL);
	ai_output = ai_svm_acc_motion_outputs_get(accelerometer_motion, NULL);
}

static void AI_Run(float *pIn, float *pOut) {
	ai_i32 batch;
	ai_error err;

	ai_input[0].data = AI_HANDLE_PTR(pIn);
	ai_output[0].data = AI_HANDLE_PTR(pOut);

	batch = ai_svm_acc_motion_run(accelerometer_motion, ai_input, ai_output);
	if (batch != 1) {
		err = ai_svm_acc_motion_get_error(accelerometer_motion);
		printf("AI ai_network_run error - type=%d code=%d\r\n", err.type,
				err.code);
		Error_Handler();
	}
}

uint8_t make_prediction(ai_float model_output) {
	return (model_output > 0) ? 1 : 0;
}

void DWT_Init(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t DWT_GetCycle(void) {
    return DWT->CYCCNT;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_I2S2_Init();
  MX_I2S3_Init();
  MX_SPI1_Init();
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN 2 */
  DWT_Init();
	BSP_ACCELERO_Init();
	AI_Init();

	uint32_t correct_predictions = 0;
	uint32_t total_predictions = 0;

	uint32_t start_cycle, end_cycle;
	ai_float cpu_mhz = SystemCoreClock / 1000000.0f;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1) {
		for (uint32_t seg = 0; seg < NUM_SEGMENTS; seg++) {
			memcpy(aiInData, test_segments[seg].features, sizeof(ai_float) * TOTAL_SAMPLES_PER_SEGMENT);

			start_cycle = DWT_GetCycle();
			AI_Run(aiInData, aiOutData);
			end_cycle = DWT_GetCycle();
			ai_float inference_time = (end_cycle - start_cycle) / cpu_mhz;

			sprintf(strTemp, "Inference time: %0.3f ms\r\n", inference_time);
			CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
			HAL_Delay(5);

			uint8_t predicted_class = make_prediction(aiOutData[0]);

			if (predicted_class == test_segments[seg].true_label) {
				correct_predictions++;
			}
			total_predictions++;

			sprintf(strPrediction, "Segment %lu, Predicted_class%d, True_class %d\r\n",
					seg, predicted_class, test_segments[seg].true_label);
			CDC_Transmit_FS((uint8_t*) strPrediction, strlen(strPrediction));
			HAL_Delay(5);

		HAL_Delay(100);

		}
		float accuracy = (float)correct_predictions / total_predictions * 100.0f;
		sprintf(strTemp, "Accuracy: %.2f%% (%lu/%lu)\r\n", accuracy, correct_predictions, total_predictions);
		CDC_Transmit_FS((uint8_t*) strTemp, strlen(strTemp));
		HAL_Delay(5);
    /* USER CODE END WHILE */
		HAL_Delay(100);
    /* USER CODE BEGIN 3 */
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 8;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_I2S;
  PeriphClkInitStruct.PLLI2S.PLLI2SN = 200;
  PeriphClkInitStruct.PLLI2S.PLLI2SM = 5;
  PeriphClkInitStruct.PLLI2S.PLLI2SR = 2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief I2S2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2S2_Init(void)
{

  /* USER CODE BEGIN I2S2_Init 0 */

  /* USER CODE END I2S2_Init 0 */

  /* USER CODE BEGIN I2S2_Init 1 */

  /* USER CODE END I2S2_Init 1 */
  hi2s2.Instance = SPI2;
  hi2s2.Init.Mode = I2S_MODE_MASTER_TX;
  hi2s2.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s2.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s2.Init.MCLKOutput = I2S_MCLKOUTPUT_DISABLE;
  hi2s2.Init.AudioFreq = I2S_AUDIOFREQ_96K;
  hi2s2.Init.CPOL = I2S_CPOL_LOW;
  hi2s2.Init.ClockSource = I2S_CLOCK_PLL;
  hi2s2.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_ENABLE;
  if (HAL_I2S_Init(&hi2s2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2S2_Init 2 */

  /* USER CODE END I2S2_Init 2 */

}

/**
  * @brief I2S3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2S3_Init(void)
{

  /* USER CODE BEGIN I2S3_Init 0 */

  /* USER CODE END I2S3_Init 0 */

  /* USER CODE BEGIN I2S3_Init 1 */

  /* USER CODE END I2S3_Init 1 */
  hi2s3.Instance = SPI3;
  hi2s3.Init.Mode = I2S_MODE_MASTER_TX;
  hi2s3.Init.Standard = I2S_STANDARD_PHILIPS;
  hi2s3.Init.DataFormat = I2S_DATAFORMAT_16B;
  hi2s3.Init.MCLKOutput = I2S_MCLKOUTPUT_ENABLE;
  hi2s3.Init.AudioFreq = I2S_AUDIOFREQ_96K;
  hi2s3.Init.CPOL = I2S_CPOL_LOW;
  hi2s3.Init.ClockSource = I2S_CLOCK_PLL;
  hi2s3.Init.FullDuplexMode = I2S_FULLDUPLEXMODE_DISABLE;
  if (HAL_I2S_Init(&hi2s3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2S3_Init 2 */

  /* USER CODE END I2S3_Init 2 */

}

/**
  * @brief SPI1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SPI1_Init(void)
{

  /* USER CODE BEGIN SPI1_Init 0 */

  /* USER CODE END SPI1_Init 0 */

  /* USER CODE BEGIN SPI1_Init 1 */

  /* USER CODE END SPI1_Init 1 */
  /* SPI1 parameter configuration*/
  hspi1.Instance = SPI1;
  hspi1.Init.Mode = SPI_MODE_MASTER;
  hspi1.Init.Direction = SPI_DIRECTION_2LINES;
  hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
  hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
  hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
  hspi1.Init.NSS = SPI_NSS_SOFT;
  hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
  hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
  hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
  hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
  hspi1.Init.CRCPolynomial = 10;
  if (HAL_SPI_Init(&hspi1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SPI1_Init 2 */

  /* USER CODE END SPI1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(CS_I2C_SPI_GPIO_Port, CS_I2C_SPI_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(OTG_FS_PowerSwitchOn_GPIO_Port, OTG_FS_PowerSwitchOn_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |Audio_RST_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : DATA_Ready_Pin */
  GPIO_InitStruct.Pin = DATA_Ready_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(DATA_Ready_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : CS_I2C_SPI_Pin */
  GPIO_InitStruct.Pin = CS_I2C_SPI_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(CS_I2C_SPI_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : INT1_Pin INT2_Pin MEMS_INT2_Pin */
  GPIO_InitStruct.Pin = INT1_Pin|INT2_Pin|MEMS_INT2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOE, &GPIO_InitStruct);

  /*Configure GPIO pin : OTG_FS_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = OTG_FS_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(OTG_FS_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PA0 */
  GPIO_InitStruct.Pin = GPIO_PIN_0;
  GPIO_InitStruct.Mode = GPIO_MODE_EVT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : LD4_Pin LD3_Pin LD5_Pin LD6_Pin
                           Audio_RST_Pin */
  GPIO_InitStruct.Pin = LD4_Pin|LD3_Pin|LD5_Pin|LD6_Pin
                          |Audio_RST_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : OTG_FS_OverCurrent_Pin */
  GPIO_InitStruct.Pin = OTG_FS_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(OTG_FS_OverCurrent_GPIO_Port, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
