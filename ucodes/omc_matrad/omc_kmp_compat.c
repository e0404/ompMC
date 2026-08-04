/******************************************************************************
 ompMC - An OpenMP parallel implementation for Monte Carlo particle transport
 simulations

 Copyright (C) 2020 Edgardo Doerner (edoerner@fis.puc.cl)


 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
*****************************************************************************/

/* Compatibility shim, compiled only when the MEX file binds to the Intel
 OpenMP runtime MATLAB ships instead of LLVM's own libomp.

 Clang emits a call to __kmpc_dispatch_deinit() after every loop with a
 dynamic or guided schedule. LLVM introduced that entry point in release 19
 so the runtime can free the loop's dispatch buffers eagerly; the runtimes
 before it -- including the Intel libiomp5md that MATLAB ships -- have no
 such call in their protocol and reclaim those buffers at thread teardown
 instead. Ignoring the call therefore reproduces exactly the behavior every
 OpenMP program had before clang 19. */
void __kmpc_dispatch_deinit(void *loc, int gtid) {

    (void)loc;
    (void)gtid;

    return;
}
