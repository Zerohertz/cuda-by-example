#include "../../include/logger.hpp"
#include "employees_salary.hpp"

void task(const double salaries[], double new_salaries[], int size)
{
    for (int i = 0; i < size; i++) {
        new_salaries[i] = salaries[i] + (salaries[i] * 15 / 100) + 5000;
    }
}

int main()
{
    const int size = sizeof(array_of_salaries) / sizeof(double);
    double    new_salaries[size];
    task(array_of_salaries, new_salaries, size);
    for (int i = 0; i < size; i++) {
        LOG_INFO(new_salaries[i]);
    }

    return 0;
}

/*
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 483470.150000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 577495.300000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 492812.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 532268.100000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 546639.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 551974.500000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 547966.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 524771.250000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 478643.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 483747.300000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 575636.900000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 526007.500000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 540177.800000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 545118.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 482751.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 548882.150000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 558346.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 513894.550000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 560087.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 489081.000000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 525909.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 552735.800000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 541947.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 514456.900000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 571424.450000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 549706.700000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 468796.150000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 542910.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 543440.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 568764.500000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 578795.950000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 548991.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 490503.550000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 565506.550000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 492081.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 546383.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 568728.850000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 549989.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 485089.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 560508.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 566833.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 488899.300000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 532560.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 480204.150000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 541042.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 495671.650000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 512865.300000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 543461.050000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 534952.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 507278.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 542437.550000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 482144.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 537652.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 573691.100000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 503502.000000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 537841.000000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 561580.450000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 499954.250000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 503668.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 574707.700000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 557615.250000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 533435.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 525700.450000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 506926.700000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 504423.150000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 553652.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 500893.800000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 470644.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 534328.900000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 532188.750000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 548674.000000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 556620.500000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 547530.900000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 499388.450000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 504353.000000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 554212.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 485957.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 523319.950000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 465399.050000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 519956.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 497738.200000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 484048.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 482381.100000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 472709.600000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 528074.050000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 556560.700000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 532906.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 538227.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 492384.950000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 501154.850000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 557739.450000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 479937.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 552142.400000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 565196.050000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 534286.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 533274.350000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 559561.050000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 469065.250000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 556691.800000
 * [2025-09-22 23:28:17] [src/lecture/chapter03/03_employees_salary.cpp:17] ℹ️ 532840.800000
 */
